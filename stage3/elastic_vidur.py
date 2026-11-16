"""Add admission, transport and VM lifetimes around unchanged Vidur replicas."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))  # This process runs in the pinned Vidur checkout.
from vidur.config import SimulationConfig
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.events.replica_schedule_event import ReplicaScheduleEvent
from vidur.scheduler.global_scheduler.global_scheduler_registry import GlobalSchedulerRegistry
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import RoundRobinGlobalScheduler
from vidur.simulator import Simulator
from vidur.types import EventType, GlobalSchedulerType
from vidur.utils.random import set_seeds


DAY_END = 86400


@dataclass
class Lease:
    replica_id: int
    started_at: float
    ready_at: float | None = None
    stop_accepting_at: float | None = None
    released_at: float | None = None
    accepted: int = 0


class ControlEvent(BaseEvent):
    def __init__(self, time, callback, *args):
        self.callback, self.args = callback, args
        super().__init__(time, EventType.GLOBAL_SCHEDULE)

    def _get_priority_number(self):
        # Update admission before arrivals and dispatch at the same instant.
        return self.time, -1, self.id

    def handle_event(self, scheduler, metrics_store):
        self.callback(self.time, *self.args)
        return []


class TransportArrival(BaseEvent):
    def __init__(self, time, replica_id, request):
        self.replica_id, self.request = replica_id, request
        super().__init__(time, EventType.REQUEST_ARRIVAL)

    def handle_event(self, scheduler, metrics_store):
        # Dispatch already admitted this request, even if the VM is now draining.
        scheduler.get_replica_scheduler(self.replica_id).add_request(self.request)
        return [ReplicaScheduleEvent(self.time, self.replica_id)]


class ElasticArrival(RequestArrivalEvent):
    def handle_event(self, scheduler, metrics_store):
        scheduler.last_arrival = self.time
        scheduler.add_request(self._request)
        metrics_store.on_request_arrival(self.time, self._request)
        return [ElasticDispatch(self.time, EventType.GLOBAL_SCHEDULE)]


class ElasticDispatch(BaseEvent):
    def handle_event(self, scheduler, metrics_store):
        immediate, transport = set(), []
        for replica_id, request in scheduler.schedule():
            scheduler.admit(self.time, replica_id, request)
            ingress = scheduler.ingress if replica_id != scheduler.local else 0
            if ingress:
                transport.append(TransportArrival(self.time + ingress, replica_id, request))
            else:
                immediate.add(replica_id)
                scheduler.get_replica_scheduler(replica_id).add_request(request)
        return transport + [ReplicaScheduleEvent(self.time, rid) for rid in immediate]


class ElasticRoundRobin(RoundRobinGlobalScheduler):
    def configure(self, simulator, plan):
        self.simulator = simulator
        self.plan = plan
        self.mode = plan["mode"]
        self.ingress, self.egress = plan["ingress_seconds"], plan["egress_seconds"]
        self.local, *self.cloud = sorted(self._replicas)
        self.ready = set(self._replicas) if self.mode == "always" else {self.local}
        self.active_leases = {}
        self.leases = []
        self.in_flight = dict.fromkeys(self._replicas, 0)
        self.routes = {}
        self.last_arrival = float("-inf")

    def enqueue(self, time, callback, *args):
        self.simulator._add_event(ControlEvent(time, callback, *args))

    def schedule(self):
        self.sort_requests()
        ready = sorted(self.ready)
        assignments = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            assignments.append((ready[self._request_counter % len(ready)], request))
            self._request_counter += 1
        return assignments

    def admit(self, time, replica_id, request):
        assert replica_id in self.ready and request.id not in self.routes
        lease = self.active_leases.get(replica_id)
        self.routes[request.id] = (replica_id, time, lease.started_at if lease else None)
        self.in_flight[replica_id] += 1
        if lease:
            lease.accepted += 1

    def start(self, time):
        for rid in self.cloud:
            assert rid not in self.active_leases
            lease = Lease(rid, time)
            self.active_leases[rid] = lease
            self.leases.append(lease)
            self.enqueue(time + self.plan["startup_seconds"], self.mark_ready, lease)

    def mark_ready(self, time, lease):
        if lease.stop_accepting_at is None:
            lease.ready_at = time
            self.ready.add(lease.replica_id)

    def stop(self, time):
        for rid, lease in list(self.active_leases.items()):
            if lease.stop_accepting_at is None:
                lease.stop_accepting_at = time
                self.ready.discard(rid)
                self.release_if_drained(time, rid)

    def release_if_drained(self, time, rid):
        lease = self.active_leases.get(rid)
        if lease and lease.stop_accepting_at is not None and self.in_flight[rid] == 0:
            lease.released_at = time
            del self.active_leases[rid]

    def finished(self, time, request_id):
        rid, _, _ = self.routes[request_id]
        if rid != self.local and self.egress:
            self.enqueue(time + self.egress, self.delivered, rid)
        else:
            self.delivered(time, rid)

    def delivered(self, time, rid):
        self.in_flight[rid] -= 1
        assert self.in_flight[rid] >= 0
        self.release_if_drained(time, rid)

    def poll(self, time):
        if time >= DAY_END:
            self.stop(time)
            return
        # Inspect the local queue using only observed enqueue time, never decode length.
        local_queue = self.get_replica_scheduler(self.local)._request_queue
        policy = self.plan["reactive"]
        if not self.active_leases and local_queue:
            oldest_wait = time - min(request.arrived_at for request in local_queue)
            if oldest_wait >= policy["oldest_local_queue_seconds"]:
                self.start(time)
        elif (self.ready - {self.local} and not local_queue
              and time - self.last_arrival >= policy["no_arrivals_for_shutdown_seconds"]):
            self.stop(time)
        self.enqueue(min(DAY_END, time + policy["poll_seconds"]), self.poll)


class ElasticSimulator(Simulator):
    def _init_event_queue(self):
        for request in self._request_generator.generate():
            self._add_event(ElasticArrival(request.arrived_at, request))

    def configure(self, plan):
        self.scheduler.configure(self, plan)
        native_end = self.metric_store._on_request_end

        def record_end(time, request):
            native_end(time, request)
            self.scheduler.finished(time, request.id)

        self.metric_store._on_request_end = record_end
        if plan["mode"] == "oracle":
            for start, stop in plan["intervals"]:
                self.scheduler.enqueue(start, self.scheduler.start)
                self.scheduler.enqueue(stop, self.scheduler.stop)
        elif plan["mode"] == "reactive":
            self.scheduler.enqueue(0, self.scheduler.poll)

    def export_elastic(self):
        scheduler = self.scheduler
        assert not scheduler.active_leases and not any(scheduler.in_flight.values())
        output = Path(self._config.metrics_config.output_dir) / "elastic.json"
        output.write_text(json.dumps(dict(local=scheduler.local, routes=scheduler.routes,
                                          leases=[asdict(lease) for lease in scheduler.leases])), encoding="utf-8")


def main():
    plan = json.loads(Path(sys.argv.pop(1)).read_text(encoding="utf-8"))
    config = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)
    key = GlobalSchedulerType.ROUND_ROBIN
    native = GlobalSchedulerRegistry.get_class(key)
    GlobalSchedulerRegistry.unregister(key)
    GlobalSchedulerRegistry.register(key, ElasticRoundRobin)
    try:
        simulator = ElasticSimulator(config)
    finally:
        GlobalSchedulerRegistry.unregister(key)
        GlobalSchedulerRegistry.register(key, native)
    simulator.configure(plan)
    simulator.run()
    simulator.export_elastic()


if __name__ == "__main__":
    main()
