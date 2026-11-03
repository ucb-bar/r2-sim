//! The section `time` of a kernel: the traffic of a task from its argument
//! block (the lane operations, and the bytes read and written per address).
//!
//! - `ops`: a list of terms, each an expression or `{expr, if: NAME}` /
//!   `{expr, unless: NAME}`; the lane operations are their sum.
//! - `reads`, `writes`: lists of `{field, bytes, if?, unless?}`: `bytes`
//!   bytes at the address in the u64 field `field` of the block, or at the
//!   base of the task's destination for `dst` (when the block has no field
//!   `dst`) and for an address of 0.
//! - An expression is a sum (`+`) of products (`*`) of integer literals and
//!   the u32 and u64 fields of the block.
//! - NAME is a flag of the kernel (a bit of its u32 field `flags`) or a u32
//!   or u64 field of the block (true when not 0).
//!
//! The names resolve to offsets at parse, so the evaluation reads the block
//! directly; its arithmetic is checked.

use serde::Deserialize;

use super::{Field, FieldType, Kernel};

/// The section as the YAML holds it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TimeEntry {
    #[serde(default)]
    ops: Vec<TermEntry>,
    #[serde(default)]
    reads: Vec<AccessEntry>,
    #[serde(default)]
    writes: Vec<AccessEntry>,
}

/// An expression as the YAML holds it: a string, or an integer literal.
#[derive(Deserialize)]
#[serde(untagged)]
enum Text {
    Str(String),
    Int(u64),
}

impl Text {
    fn text(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Int(v) => v.to_string(),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TermEntry {
    Plain(Text),
    Cond(CondTerm),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CondTerm {
    expr: Text,
    #[serde(rename = "if")]
    if_: Option<String>,
    unless: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccessEntry {
    field: String,
    bytes: Text,
    #[serde(rename = "if")]
    if_: Option<String>,
    unless: Option<String>,
}

/// The traffic of a kernel, resolved against its block.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Time {
    pub ops: Vec<Term>,
    pub reads: Vec<Access>,
    pub writes: Vec<Access>,
}

/// A term of the lane operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term {
    pub expr: Expr,
    pub cond: Option<Cond>,
}

/// An entry of `reads` or `writes`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Access {
    pub address: Address,
    pub bytes: Expr,
    pub cond: Option<Cond>,
}

/// The address of an access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address {
    /// A u64 field of the block (its value; 0: the destination).
    Field { name: String, offset: usize },
    /// The base of the task's destination.
    Dst,
}

/// The condition of a term or an access: `if: name` or `unless: name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cond {
    pub name: String,
    pub unless: bool,
    test: Test,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Test {
    /// A bit of the u32 field at `offset` (`flags`).
    Flag { bit: u32, offset: usize },
    /// An integer field, true when not 0.
    Field(Value),
}

/// An integer field of the block: its offset, and whether it is a u64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Value {
    offset: usize,
    wide: bool,
}

impl Value {
    fn read(self, block: &[u8]) -> u64 {
        let at = |n: usize| &block[self.offset..self.offset + n];
        if self.wide {
            u64::from_le_bytes(at(8).try_into().expect("8 bytes"))
        } else {
            u32::from_le_bytes(at(4).try_into().expect("4 bytes")) as u64
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Factor {
    Literal(u64),
    Field(Value),
}

/// A sum of products of literals and integer fields, with its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    pub text: String,
    terms: Vec<Vec<Factor>>,
}

impl Expr {
    /// The value on `block`; `None` on an overflow.
    fn eval(&self, block: &[u8]) -> Option<u64> {
        self.terms.iter().try_fold(0u64, |sum, product| {
            let p = product.iter().try_fold(1u64, |p, f| {
                p.checked_mul(match *f {
                    Factor::Literal(v) => v,
                    Factor::Field(v) => v.read(block),
                })
            })?;
            sum.checked_add(p)
        })
    }
}

impl Cond {
    fn holds(&self, block: &[u8]) -> bool {
        let set = match self.test {
            Test::Flag { bit, offset } => {
                Value {
                    offset,
                    wide: false,
                }
                .read(block)
                    & bit as u64
                    != 0
            }
            Test::Field(v) => v.read(block) != 0,
        };
        set != self.unless
    }
}

/// The traffic of one task: its lane operations, and (address, bytes) per
/// read and per write.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Traffic {
    pub ops: u64,
    pub reads: Vec<(u64, u64)>,
    pub writes: Vec<(u64, u64)>,
}

/// The integer field `name` of `fields`, or the reason it is not one.
fn value(fields: &[Field], name: &str) -> Result<Value, String> {
    let f = fields
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| format!("unknown field {name}"))?;
    match f.ty {
        FieldType::F32 => Err(format!("field {name} is f32, not an integer field")),
        ty => Ok(Value {
            offset: f.offset as usize,
            wide: ty == FieldType::U64,
        }),
    }
}

fn expr(text: &Text, fields: &[Field]) -> Result<Expr, String> {
    let text = text.text();
    let mut terms = Vec::new();
    for term in text.split('+') {
        let mut product = Vec::new();
        for f in term.split('*').map(str::trim) {
            if f.is_empty() {
                return Err(format!("an empty operand in {text:?}"));
            }
            let factor = if f.bytes().all(|b| b.is_ascii_digit()) {
                Factor::Literal(
                    f.parse()
                        .map_err(|_| format!("the literal {f} in {text:?} overflows u64"))?,
                )
            } else {
                Factor::Field(value(fields, f).map_err(|e| format!("{e} in {text:?}"))?)
            };
            product.push(factor);
        }
        terms.push(product);
    }
    Ok(Expr { text, terms })
}

fn cond(
    if_: &Option<String>,
    unless: &Option<String>,
    kernel: &Kernel,
) -> Result<Option<Cond>, String> {
    let (name, negate) = match (if_, unless) {
        (Some(_), Some(_)) => return Err("both if and unless in one entry".into()),
        (Some(n), None) => (n, false),
        (None, Some(n)) => (n, true),
        (None, None) => return Ok(None),
    };
    let test = if let Some(&bit) = kernel.flags.get(name) {
        let flags = kernel
            .field("flags")
            .filter(|f| f.ty == FieldType::U32)
            .ok_or_else(|| format!("flag {name}: the block has no u32 field flags"))?;
        Test::Flag {
            bit,
            offset: flags.offset as usize,
        }
    } else if kernel.field(name).is_some() {
        Test::Field(value(&kernel.fields, name)?)
    } else {
        return Err(format!(
            "{name} is neither a flag of the kernel nor a field of the block"
        ));
    };
    Ok(Some(Cond {
        name: name.clone(),
        unless: negate,
        test,
    }))
}

fn access(e: &AccessEntry, kernel: &Kernel) -> Result<Access, String> {
    let address = match kernel.field(&e.field) {
        Some(f) if f.ty == FieldType::U64 => Address::Field {
            name: f.name.clone(),
            offset: f.offset as usize,
        },
        None if e.field == "dst" => Address::Dst,
        _ => {
            return Err(format!(
                "field {}: not a u64 field of the block and not dst",
                e.field
            ))
        }
    };
    Ok(Access {
        address,
        bytes: expr(&e.bytes, &kernel.fields)?,
        cond: cond(&e.if_, &e.unless, kernel)?,
    })
}

/// The section `e` of `kernel` (its fields and flags parsed).
pub(super) fn compile(e: &TimeEntry, kernel: &Kernel) -> Result<Time, String> {
    let ops = e
        .ops
        .iter()
        .map(|t| {
            Ok(match t {
                TermEntry::Plain(text) => Term {
                    expr: expr(text, &kernel.fields)?,
                    cond: None,
                },
                TermEntry::Cond(c) => Term {
                    expr: expr(&c.expr, &kernel.fields)?,
                    cond: cond(&c.if_, &c.unless, kernel)?,
                },
            })
        })
        .collect::<Result<_, String>>()?;
    let list = |l: &[AccessEntry]| {
        l.iter()
            .map(|a| access(a, kernel))
            .collect::<Result<Vec<_>, String>>()
    };
    Ok(Time {
        ops,
        reads: list(&e.reads)?,
        writes: list(&e.writes)?,
    })
}

impl Kernel {
    /// The traffic of a task of this kernel with the argument block
    /// `block`, whose destination starts at `dst`. An address field of 0,
    /// and `dst`, resolve to `dst`. Entries whose condition is false or
    /// whose bytes are 0 are left out.
    pub fn traffic(&self, block: &[u8], dst: u64) -> Result<Traffic, String> {
        if block.len() as u64 != self.bytes {
            return Err(format!(
                "kernel {} (id {}): an argument block of {} bytes, not {}",
                self.name,
                self.id,
                block.len(),
                self.bytes
            ));
        }
        let eval = |e: &Expr| {
            e.eval(block).ok_or_else(|| {
                format!(
                    "kernel {} (id {}): time: {:?} overflows u64",
                    self.name, self.id, e.text
                )
            })
        };
        let on = |c: &Option<Cond>| c.as_ref().is_none_or(|c| c.holds(block));
        let mut ops = 0u64;
        for t in self.time.ops.iter().filter(|t| on(&t.cond)) {
            ops = ops.checked_add(eval(&t.expr)?).ok_or_else(|| {
                format!(
                    "kernel {} (id {}): time: the ops overflow u64",
                    self.name, self.id
                )
            })?;
        }
        let entries = |list: &[Access]| -> Result<Vec<(u64, u64)>, String> {
            let mut out = Vec::new();
            for a in list.iter().filter(|a| on(&a.cond)) {
                let bytes = eval(&a.bytes)?;
                if bytes == 0 {
                    continue;
                }
                let addr = match a.address {
                    Address::Dst => dst,
                    Address::Field { offset, .. } => {
                        match (Value { offset, wide: true }).read(block) {
                            0 => dst,
                            v => v,
                        }
                    }
                };
                out.push((addr, bytes));
            }
            Ok(out)
        };
        Ok(Traffic {
            ops,
            reads: entries(&self.time.reads)?,
            writes: entries(&self.time.writes)?,
        })
    }
}
