//! Minimal slab: stable indices, O(1) insert and remove, no dependencies.

#[derive(Debug, Clone)]
pub(crate) struct Slab<T> {
    slots: Vec<Option<T>>,
    free: Vec<usize>,
    len: usize,
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }
}

impl<T> Slab<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn insert(&mut self, value: T) -> usize {
        self.len += 1;
        match self.free.pop() {
            Some(i) => {
                self.slots[i] = Some(value);
                i
            }
            None => {
                self.slots.push(Some(value));
                self.slots.len() - 1
            }
        }
    }

    pub fn remove(&mut self, i: usize) -> Option<T> {
        let v = self.slots.get_mut(i)?.take();
        if v.is_some() {
            self.len -= 1;
            self.free.push(i);
        }
        v
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.slots.get(i).and_then(|s| s.as_ref())
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.slots.get_mut(i).and_then(|s| s.as_mut())
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|v| (i, v)))
    }
}
