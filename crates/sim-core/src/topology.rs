//! The hardware tree and the fabric (architecture Sections 3.1, 3.5 and 8).
//!
//! The tree has one node per Neutrino. The fabric is a graph of buses:
//! every bus is a vertex, and a **link** (a `topology: link` fabric entry,
//! architecture Section 8.2) is an edge between two buses: usually a
//! child's bus to its parent's bus, or a cache that is a slave on one bus
//! and a master on another. A bus is a data bus or a control bus; data
//! links join data buses and control links control buses, so data and
//! control traffic never share a resource. Memories, a node's Neutrino,
//! compute masters and each end of a link are **attachments** of a bus;
//! transfers go from one attachment to another. A compute master's
//! attachment has no port.
//!
//! What a transfer charges on a bus depends on its topology (Section 3.5):
//! an `xbar` has per attachment an ingress and an egress resource, and a
//! transfer from attachment s to attachment d charges s's ingress and d's
//! egress; a `bus` has one shared resource. A bus without a rate is
//! unlimited. A link keeps one resource per direction (full duplex). A
//! bus or link with `wr_bytes` has a second, write channel beside each of
//! these (Section 3.5): a write moves its data over the write channel, a
//! read over the read channel, and a route has a resource list per
//! channel ([`Path::hops_for`]).
//!
//! A route is the path with the fewest links, found when the fabric is
//! finished. Its latency is the sum of the latencies of every bus and
//! link on it. Between sets of attachments (a memory attached to two
//! buses, a master of two buses) the route with the fewest links wins;
//! a tie goes to the first pair in attachment order.

use std::collections::VecDeque;

use nu_model::NeutrinoId;

use crate::resource::{Dir, ResourceId};
use crate::time::Cycle;

/// One node of the hardware tree: one Neutrino.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopoNode {
    /// The node name from the configuration.
    pub name: String,
    /// The parent node; `None` for a root.
    pub parent: Option<NeutrinoId>,
    /// The child nodes, in insertion order.
    pub children: Vec<NeutrinoId>,
    /// Distance from the root; 0 for a root.
    pub depth: u32,
}

/// Identifies a bus of the fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BusId(pub u32);

/// Identifies an attachment (a port of a bus).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AttachId(pub u32);

/// The resource of one direction of a port, a bus or a link; `None` is
/// unlimited.
pub type Lane = Option<ResourceId>;

/// How a bus charges a transfer (Section 3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusTopology {
    /// Per attachment an ingress and an egress resource.
    Xbar,
    /// One shared resource.
    Shared,
}

/// A bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bus {
    /// Dotted name from the root.
    pub name: String,
    /// How the bus charges.
    pub topology: BusTopology,
    /// The shared lane (`Shared` only).
    pub lane: Lane,
    /// Hop latency in cycles.
    pub latency: Cycle,
    /// The shared write lane (`Shared` with `wr_bytes`).
    pub lane_wr: Lane,
    /// A control bus (protocol messages); otherwise a data bus.
    pub control: bool,
    /// Transaction size: a transfer through the bus is cut at multiples
    /// of it (architecture Section 5.3).
    pub block: Option<u64>,
}

/// One attachment of a bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment {
    /// The bus.
    pub bus: BusId,
    /// Dotted name (`<bus>.<attachment>`).
    pub name: String,
    /// Into the bus from this attachment (`Xbar` only).
    pub ingress: Lane,
    /// Out of the bus to this attachment (`Xbar` only).
    pub egress: Lane,
    /// The write channel's ingress and egress (`wr_bytes`).
    pub ingress_wr: Lane,
    pub egress_wr: Lane,
}

/// A link between buses `a` (its `from`) and `b` (its `to`), attached to
/// them as `att_a` and `att_b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FabricLink {
    pub a: BusId,
    pub b: BusId,
    pub att_a: AttachId,
    pub att_b: AttachId,
    /// Lanes from `a` to `b`, and from `b` to `a`.
    pub ab: Lane,
    pub ba: Lane,
    /// The write channel of each direction (`wr_bytes`).
    pub ab_wr: Lane,
    pub ba_wr: Lane,
    pub latency: Cycle,
    /// Transaction size, as [`Bus::block`].
    pub block: Option<u64>,
}

/// One link of a route, rated or unlimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Link {
    /// The ingress port of an `Xbar` attachment.
    In(AttachId),
    /// The egress port of an `Xbar` attachment.
    Out(AttachId),
    /// The shared lane of a `Shared` bus.
    Bus(BusId),
    /// One direction of a fabric link: its index, and true for `a` to `b`
    /// (from to to).
    Fabric(u32, bool),
}

/// One hop of a control route: a resource and the latency after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hop {
    /// The resource.
    pub lane: ResourceId,
    /// Cycles from the end of the transmission to the next stage.
    pub latency: Cycle,
}

/// The resources and latency of one route.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Path {
    /// The resources the route charges, in route order.
    pub resources: Vec<ResourceId>,
    /// Sum of the hop latencies in cycles.
    pub latency: Cycle,
    /// Control routes: the hops with a lane, in route order. The latency
    /// of a hop without a lane (unlimited) is added to the hop before it,
    /// or to `lead` when no hop comes before it.
    pub hops: Vec<Hop>,
    /// Control routes: latency before the first hop.
    pub lead: Cycle,
    /// Links on the route.
    pub links: u32,
    /// The distinct `block_bytes` of the buses and links on the route.
    pub blocks: Vec<u64>,
    /// The write channel's resources and hops, when a bus or link of the
    /// route has one; `None`: the read channel's.
    pub write: Option<Box<(Vec<ResourceId>, Vec<Hop>)>>,
}

/// The bytes from address `a` to the next multiple of each of `blocks`:
/// the most a chunk at `a` may move (`u64::MAX` without blocks).
pub fn grid_run(a: u64, blocks: &[u64]) -> u64 {
    blocks.iter().map(|b| b - a % b).min().unwrap_or(u64::MAX)
}

impl Path {
    /// Appends a hop of `lane` (none: unlimited) and `latency`.
    fn push_hop(&mut self, lane: Option<ResourceId>, latency: Cycle) {
        match (lane, self.hops.last_mut()) {
            (Some(lane), _) => self.hops.push(Hop { lane, latency }),
            (None, Some(h)) => h.latency += latency,
            (None, None) => self.lead += latency,
        }
    }

    /// The resources the route charges for data moving in direction
    /// `dir`: a write on the write channels.
    pub fn resources_for(&self, dir: Dir) -> &[ResourceId] {
        match (&self.write, dir) {
            (Some(w), Dir::Write) => &w.0,
            _ => &self.resources,
        }
    }

    /// The hops of the route for data moving in direction `dir` (a hop
    /// rated on one channel is rated on the other, so the latencies and
    /// `lead` are the same).
    pub fn hops_for(&self, dir: Dir) -> &[Hop] {
        match (&self.write, dir) {
            (Some(w), Dir::Write) => &w.1,
            _ => &self.hops,
        }
    }
}

/// The hardware tree and the fabric.
#[derive(Debug, Clone, Default)]
pub struct Topology {
    nodes: Vec<TopoNode>,
    /// The attachment of a node's Neutrino on its control bus (messages).
    hub: Vec<Option<AttachId>>,
    /// The attachments of a node's Neutrino on its data buses (loads and
    /// stores).
    data_hub: Vec<Vec<AttachId>>,
    buses: Vec<Bus>,
    attachments: Vec<Attachment>,
    links: Vec<FabricLink>,
    /// Routes by (from, to) attachment, after `finish`; `None` where no
    /// route exists.
    routes: Vec<Option<Path>>,
    /// Bus-level routes by (from, to) bus, after `finish`.
    steps: Vec<Option<Vec<Step>>>,
    finished: bool,
}

static EMPTY: Path = Path {
    resources: Vec::new(),
    latency: 0,
    hops: Vec::new(),
    lead: 0,
    links: 0,
    blocks: Vec::new(),
    write: None,
};

/// One bus of a bus-level route and the link (index, direction a to b)
/// that reached it.
type Step = (usize, Option<(usize, bool)>);

impl Topology {
    /// Adds a node; ids are dense and follow insertion order.
    pub(crate) fn push(&mut self, name: &str, parent: Option<NeutrinoId>) {
        let id = NeutrinoId(self.nodes.len() as u32);
        let depth = match parent {
            Some(p) => {
                self.nodes[p.0 as usize].children.push(id);
                self.nodes[p.0 as usize].depth + 1
            }
            None => 0,
        };
        self.nodes.push(TopoNode {
            name: name.to_string(),
            parent,
            children: Vec::new(),
            depth,
        });
        self.hub.push(None);
        self.data_hub.push(Vec::new());
    }

    pub(crate) fn add_bus(&mut self, bus: Bus) -> BusId {
        self.finished = false;
        self.buses.push(bus);
        BusId(self.buses.len() as u32 - 1)
    }

    pub(crate) fn add_attachment(&mut self, att: Attachment) -> AttachId {
        self.finished = false;
        self.attachments.push(att);
        AttachId(self.attachments.len() as u32 - 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_link(
        &mut self,
        att_a: AttachId,
        att_b: AttachId,
        ab: Lane,
        ba: Lane,
        latency: Cycle,
    ) {
        self.finished = false;
        let (a, b) = (self.attachment(att_a).bus, self.attachment(att_b).bus);
        self.links.push(FabricLink {
            a,
            b,
            att_a,
            att_b,
            ab,
            ba,
            ab_wr: None,
            ba_wr: None,
            latency,
            block: None,
        });
    }

    pub(crate) fn set_link_wr(&mut self, link: usize, ab: Lane, ba: Lane) {
        self.finished = false;
        let l = &mut self.links[link];
        (l.ab_wr, l.ba_wr) = (ab, ba);
    }

    pub(crate) fn set_bus_wr(&mut self, bus: BusId, lane: Lane) {
        self.finished = false;
        self.buses[bus.0 as usize].lane_wr = lane;
    }

    pub(crate) fn set_bus_block(&mut self, bus: BusId, block: u64) {
        self.finished = false;
        self.buses[bus.0 as usize].block = Some(block);
    }

    pub(crate) fn set_link_block(&mut self, link: usize, block: u64) {
        self.finished = false;
        self.links[link].block = Some(block);
    }

    pub(crate) fn set_hub(&mut self, nu: NeutrinoId, control: AttachId, data: Vec<AttachId>) {
        self.hub[nu.0 as usize] = Some(control);
        self.data_hub[nu.0 as usize] = data;
    }

    /// Computes every route. Fails when two routes between two buses have
    /// the same, fewest number of hops.
    pub(crate) fn finish(&mut self) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        let nb = self.buses.len();
        let mut adj: Vec<Vec<(usize, usize, bool)>> = vec![Vec::new(); nb];
        for (i, br) in self.links.iter().enumerate() {
            adj[br.a.0 as usize].push((br.b.0 as usize, i, true));
            adj[br.b.0 as usize].push((br.a.0 as usize, i, false));
        }
        // Bus-level routes: the buses and the links that reach them.
        let mut steps: Vec<Option<Vec<Step>>> = vec![None; nb * nb];
        for from in 0..nb {
            let mut hops = vec![usize::MAX; nb];
            let mut count = vec![0u32; nb];
            let mut prev: Vec<Option<(usize, usize, bool)>> = vec![None; nb];
            hops[from] = 0;
            count[from] = 1;
            let mut q = VecDeque::from([from]);
            while let Some(u) = q.pop_front() {
                for &(v, br, fwd) in &adj[u] {
                    if hops[v] == usize::MAX {
                        hops[v] = hops[u] + 1;
                        prev[v] = Some((u, br, fwd));
                        q.push_back(v);
                    }
                    if hops[v] == hops[u] + 1 {
                        count[v] = count[v].saturating_add(count[u]);
                    }
                }
            }
            for to in 0..nb {
                if count[to] > 1 {
                    return Err(format!(
                        "two routes of {} hops from {} to {}",
                        hops[to], self.buses[from].name, self.buses[to].name
                    ));
                }
                if hops[to] == usize::MAX {
                    continue;
                }
                let mut route = Vec::new();
                let mut cur = to;
                while let Some((u, br, fwd)) = prev[cur] {
                    route.push((cur, Some((br, fwd))));
                    cur = u;
                }
                route.push((from, None));
                route.reverse();
                steps[from * nb + to] = Some(route);
            }
        }
        let na = self.attachments.len();
        self.routes = Vec::with_capacity(na * na);
        for s in 0..na {
            for d in 0..na {
                let (sb, db) = (
                    self.attachments[s].bus.0 as usize,
                    self.attachments[d].bus.0 as usize,
                );
                let p = steps[sb * nb + db]
                    .as_ref()
                    .map(|route| self.route_of(s, d, route));
                self.routes.push(p);
            }
        }
        self.steps = steps;
        self.finished = true;
        Ok(())
    }

    /// Calls `f` for every link of the route from attachment `s` to `d`
    /// over the bus-level `route`, in route order, with its lane, its
    /// write lane (the lane without a write channel) and the latency after
    /// it.
    fn walk(&self, s: usize, d: usize, route: &[Step], mut f: impl FnMut(Link, Lane, Lane, Cycle)) {
        for (i, &(bus, via)) in route.iter().enumerate() {
            // Where the transfer enters and leaves this bus.
            let entry = match via {
                None => s,
                Some((br, fwd)) => {
                    let b = &self.links[br];
                    let (lane, wr) = if fwd {
                        (b.ab, b.ab_wr)
                    } else {
                        (b.ba, b.ba_wr)
                    };
                    f(Link::Fabric(br as u32, fwd), lane, wr.or(lane), b.latency);
                    (if fwd { b.att_b } else { b.att_a }).0 as usize
                }
            };
            let exit = match route.get(i + 1) {
                None => d,
                Some(&(_, Some((br, fwd)))) => {
                    let b = &self.links[br];
                    (if fwd { b.att_a } else { b.att_b }).0 as usize
                }
                Some(&(_, None)) => unreachable!("only the first step has no link"),
            };
            let b = &self.buses[bus];
            match b.topology {
                BusTopology::Xbar => {
                    let (i, o) = (&self.attachments[entry], &self.attachments[exit]);
                    f(
                        Link::In(AttachId(entry as u32)),
                        i.ingress,
                        i.ingress_wr.or(i.ingress),
                        0,
                    );
                    f(
                        Link::Out(AttachId(exit as u32)),
                        o.egress,
                        o.egress_wr.or(o.egress),
                        b.latency,
                    );
                }
                BusTopology::Shared => f(
                    Link::Bus(BusId(bus as u32)),
                    b.lane,
                    b.lane_wr.or(b.lane),
                    b.latency,
                ),
            }
        }
    }

    /// Every link of the route from attachment `a` to `b` with its lane,
    /// in route order (empty when there is no route).
    pub fn links(&self, a: AttachId, b: AttachId) -> Vec<(Link, Lane)> {
        assert!(self.finished, "fabric routes used before finish");
        let (sb, db) = (self.attachment(a).bus.0, self.attachment(b).bus.0);
        let nb = self.buses.len();
        let mut out = Vec::new();
        if let Some(route) = &self.steps[sb as usize * nb + db as usize] {
            self.walk(a.0 as usize, b.0 as usize, route, |l, lane, _, _| {
                out.push((l, lane))
            });
        }
        out
    }

    /// The path from attachment `s` to attachment `d` over the bus-level
    /// `route`.
    fn route_of(&self, s: usize, d: usize, route: &[Step]) -> Path {
        let mut p = Path {
            links: route.len() as u32 - 1,
            ..Path::default()
        };
        let mut wr = Path::default();
        self.walk(s, d, route, |_, lane, lane_wr, latency| {
            p.resources.extend(lane);
            p.push_hop(lane, latency);
            p.latency += latency;
            wr.resources.extend(lane_wr);
            wr.push_hop(lane_wr, latency);
        });
        if wr.resources != p.resources {
            p.write = Some(Box::new((wr.resources, wr.hops)));
        }
        for &(bus, via) in route {
            let link = via.and_then(|(l, _)| self.links[l].block);
            for b in [link, self.buses[bus].block].into_iter().flatten() {
                if !p.blocks.contains(&b) {
                    p.blocks.push(b);
                }
            }
        }
        p
    }

    /// The route from attachment `a` to attachment `b`, if one exists.
    pub fn route(&self, a: AttachId, b: AttachId) -> Option<&Path> {
        assert!(self.finished, "fabric routes used before finish");
        self.routes[a.0 as usize * self.attachments.len() + b.0 as usize].as_ref()
    }

    /// The route with the fewest links from any of `from` to any of `to`;
    /// a tie goes to the first pair in order. Empty when either set is
    /// empty; `None` when no pair has a route.
    pub fn best_path(&self, from: &[AttachId], to: &[AttachId]) -> Option<&Path> {
        if from.is_empty() || to.is_empty() {
            return Some(&EMPTY);
        }
        self.best_pair(from, to)
            .map(|(a, b)| self.route(a, b).expect("route"))
    }

    /// The (from, to) pair of [`Topology::best_path`]; `None` when either
    /// set is empty or no pair has a route.
    pub fn best_pair(&self, from: &[AttachId], to: &[AttachId]) -> Option<(AttachId, AttachId)> {
        let mut best: Option<(AttachId, AttachId, u32)> = None;
        for &a in from {
            for &b in to {
                if let Some(p) = self.route(a, b) {
                    if best.is_none_or(|q| p.links < q.2) {
                        best = Some((a, b, p.links));
                    }
                }
            }
        }
        best.map(|(a, b, _)| (a, b))
    }

    /// A copy for [`Topology::links`] only: without the attachment routes.
    pub fn links_only(&self) -> Topology {
        Topology {
            nodes: self.nodes.clone(),
            hub: self.hub.clone(),
            data_hub: self.data_hub.clone(),
            buses: self.buses.clone(),
            attachments: self.attachments.clone(),
            links: self.links.clone(),
            routes: Vec::new(),
            steps: self.steps.clone(),
            finished: self.finished,
        }
    }

    /// Number of buses.
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Number of attachments.
    pub fn attachment_count(&self) -> usize {
        self.attachments.len()
    }

    /// The fabric links, in insertion order.
    pub fn fabric_links(&self) -> &[FabricLink] {
        &self.links
    }

    /// The largest latency of any route (0 before `finish`).
    pub fn max_latency(&self) -> Cycle {
        self.routes
            .iter()
            .flatten()
            .map(|p| p.latency)
            .max()
            .unwrap_or(0)
    }

    /// The node of Neutrino `id`; panics if `id` is unknown.
    pub fn node(&self, id: NeutrinoId) -> &TopoNode {
        &self.nodes[id.0 as usize]
    }

    /// The attachment of Neutrino `id` on its control bus (messages).
    pub fn hub(&self, id: NeutrinoId) -> Option<AttachId> {
        self.hub[id.0 as usize]
    }

    /// The attachments of Neutrino `id` on its data buses (loads and
    /// stores).
    pub fn data_hub(&self, id: NeutrinoId) -> &[AttachId] {
        &self.data_hub[id.0 as usize]
    }

    /// The bus `id`.
    pub fn bus(&self, id: BusId) -> &Bus {
        &self.buses[id.0 as usize]
    }

    /// The attachment `id`.
    pub fn attachment(&self, id: AttachId) -> &Attachment {
        &self.attachments[id.0 as usize]
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the tree has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Dotted name from the root ("root.gpc0.sm1"): the prefix of the
    /// node's resource, memory and PE names.
    pub fn path_name(&self, id: NeutrinoId) -> String {
        let mut names = vec![self.node(id).name.as_str()];
        let mut cur = id;
        while let Some(p) = self.node(cur).parent {
            names.push(self.node(p).name.as_str());
            cur = p;
        }
        names.reverse();
        names.join(".")
    }

    /// The data route between the Neutrinos of nodes `a` and `b`.
    pub fn path(&self, a: NeutrinoId, b: NeutrinoId) -> &Path {
        self.best_path(self.data_hub(a), self.data_hub(b))
            .unwrap_or(&EMPTY)
    }

    /// The control route between the Neutrinos of nodes `a` and `b`
    /// (messages).
    pub(crate) fn control_path(&self, a: NeutrinoId, b: NeutrinoId) -> &Path {
        match (self.hub(a), self.hub(b)) {
            (Some(x), Some(y)) => self.route(x, y).unwrap_or(&EMPTY),
            _ => &EMPTY,
        }
    }
}
