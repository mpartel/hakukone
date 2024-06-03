use std::ops::{Deref, DerefMut};

/// Like `Option`, but implements `Deref` that panics when the element is None.
/// Useful for frequently used values that are almost never None, to reduce clutter around most accesses.
pub enum DerefOption<T> {
    /// A value.
    Some(T),
    /// No value.
    None,
}

impl<T> DerefOption<T> {
    pub fn is_some(&self) -> bool {
        matches!(self, DerefOption::Some(_))
    }
    pub fn is_none(&self) -> bool {
        matches!(self, DerefOption::None)
    }
}

impl<T> Deref for DerefOption<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            DerefOption::Some(ref x) => x,
            DerefOption::None => panic!("null value deref'ed"),
        }
    }
}

impl<T> DerefMut for DerefOption<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            DerefOption::Some(ref mut x) => x,
            DerefOption::None => panic!("null value deref'ed"),
        }
    }
}

impl<T> From<T> for DerefOption<T> {
    fn from(value: T) -> Self {
        DerefOption::Some(value)
    }
}

impl<T> From<DerefOption<T>> for Option<T> {
    fn from(value: DerefOption<T>) -> Self {
        match value {
            DerefOption::Some(x) => Some(x),
            DerefOption::None => None,
        }
    }
}
