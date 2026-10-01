//! The hardware tree and the fabric (architecture Sections 3.1, 3.5 and 8).
//!
//! The tree has one node per Neutrino. The fabric is a graph of buses:
//! every bus is a vertex, and a bridge is an edge between two buses (one
//! side of a child's bridge to a parent bus, or a cache that is a slave on
//! one bus and a master on another). A bus is a data bus or a control bus;
//! a child's bridge joins its data bus to the parent's data bus and its
//! control bus to the parent's control bus, so data and control traffic
//! never share a resource. Memories, a node's Neutrino, compute masters and
//! each end of a bridge are **attachments** of a bus; transfers go from one
//! attachment to another. A compute master's attachment has no port.
//!
//! What a transfer charges on a bus depends on its topology (Section 3.5):
//! an `xbar` has per attachment an ingress and an egress resource, and a
//! transfer from attachment s to attachment d charges s's ingress and d's
//! egress; a `bus` has one shared resource. A bus without a rate is
//! unlimited. A bridge keeps one resource per direction.
//!
//! A route is the path with the fewest bridges, found when the fabric is
//! finished. Its latency is the sum of the latencies of every bus and
//! bridge on it. Between sets of attachments (a memory attached to two
//! buses, a master of two buses) the route with the fewest bridges wins;
//! a tie goes to the first pair in attachment order.

use std::collections::VecDeque;

use nu_model::NeutrinoId;

use crate::resource::ResourceId;
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

/// The resource of one direction of a port, a bus or a bridge; `None` is
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
    /// A control bus (protocol messages); otherwise a data bus.
    pub control: bool,
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
}

/// A bridge between buses `a` and `b`, attached to them as `att_a` and
/// `att_b`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Bridge {
    a: BusId,
    b: BusId,
    att_a: AttachId,
    att_b: AttachId,
    /// Lanes from `a` to `b`, and from `b` to `a`.
    ab: Lane,
    ba: Lane,
    latency: Cycle,
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
    /// Bridges on the route.
    pub bridges: u32,
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
    bridges: Vec<Bridge>,
    /// Routes by (from, to) attachment, after `finish`; `None` where no
    /// route exists.
    routes: Vec<Option<Path>>,
    finished: bool,
}

static EMPTY: Path = Path {
    resources: Vec::new(),
    latency: 0,
    hops: Vec::new(),
    lead: 0,
    bridges: 0,
};

/// One bus of a bus-level route and the bridge (index, direction a to b)
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
    pub(crate) fn add_bridge(
        &mut self,
        att_a: AttachId,
        att_b: AttachId,
        ab: Lane,
        ba: Lane,
        latency: Cycle,
    ) {
        self.finished = false;
        let (a, b) = (self.attachment(att_a).bus, self.attachment(att_b).bus);
        self.bridges.push(Bridge {
            a,
            b,
            att_a,
            att_b,
            ab,
            ba,
            latency,
        });
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
        for (i, br) in self.bridges.iter().enumerate() {
            adj[br.a.0 as usize].push((br.b.0 as usize, i, true));
            adj[br.b.0 as usize].push((br.a.0 as usize, i, false));
        }
        // Bus-level routes: the buses and the bridges that reach them.
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
        self.finished = true;
        Ok(())
    }

    /// The path from attachment `s` to attachment `d` over the bus-level
    /// `route`.
    fn route_of(&self, s: usize, d: usize, route: &[Step]) -> Path {
        let mut p = Path {
            bridges: route.len() as u32 - 1,
            ..Path::default()
        };
        let mut add = |lane: Lane, latency: Cycle| {
            p.resources.extend(lane);
            p.push_hop(lane, latency);
        };
        let mut latency = 0;
        for (i, &(bus, via)) in route.iter().enumerate() {
            // Where the transfer enters and leaves this bus.
            let entry = match via {
                None => s,
                Some((br, fwd)) => {
                    let b = &self.bridges[br];
                    (if fwd { b.att_b } else { b.att_a }).0 as usize
                }
            };
            if let Some((br, fwd)) = via {
                let b = &self.bridges[br];
                add(if fwd { b.ab } else { b.ba }, b.latency);
                latency += b.latency;
            }
            let exit = match route.get(i + 1) {
                None => d,
                Some(&(_, Some((br, fwd)))) => {
                    let b = &self.bridges[br];
                    (if fwd { b.att_a } else { b.att_b }).0 as usize
                }
                Some(&(_, None)) => unreachable!("only the first step has no bridge"),
            };
            let b = &self.buses[bus];
            match b.topology {
                BusTopology::Xbar => {
                    add(self.attachments[entry].ingress, 0);
                    add(self.attachments[exit].egress, b.latency);
                }
                BusTopology::Shared => add(b.lane, b.latency),
            }
            latency += b.latency;
        }
        p.latency = latency;
        p
    }

    /// The route from attachment `a` to attachment `b`, if one exists.
    pub fn route(&self, a: AttachId, b: AttachId) -> Option<&Path> {
        assert!(self.finished, "fabric routes used before finish");
        self.routes[a.0 as usize * self.attachments.len() + b.0 as usize].as_ref()
    }

    /// The route with the fewest bridges from any of `from` to any of `to`;
    /// a tie goes to the first pair in order. Empty when either set is
    /// empty; `None` when no pair has a route.
    pub fn best_path(&self, from: &[AttachId], to: &[AttachId]) -> Option<&Path> {
        if from.is_empty() || to.is_empty() {
            return Some(&EMPTY);
        }
        let mut best: Option<&Path> = None;
        for &a in from {
            for &b in to {
                if let Some(p) = self.route(a, b) {
                    if best.is_none_or(|q| p.bridges < q.bridges) {
                        best = Some(p);
                    }
                }
            }
        }
        best
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
