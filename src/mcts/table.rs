//! 稀疏表：固定长度的 Vec<Option<T>> 封装。
//!
//! 用于按棋盘位置索引的子节点表。每个槽位要么有值要么为空，
//! 提供位置→值的 O(1) 查找和方便的非空迭代。

use std::ops::{Index, IndexMut};

/// 固定长度的稀疏表。
///
/// 内部为 `Vec<Option<T>>`，通过 `Index<usize>` 访问槽位。
#[derive(Clone)]
pub struct Table<T> {
    slots: Vec<Option<T>>,
}

impl<T> Table<T> {
    /// 创建长度为 `size` 的空表。
    pub fn new(size: usize) -> Self {
        Self {
            slots: std::iter::repeat_with(|| None).take(size).collect(),
        }
    }

    /// 返回表的长度（槽位总数）。
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// 设置 `idx` 位置的值为 `value`。
    #[inline]
    pub fn set(&mut self, idx: usize, value: T) {
        self.slots[idx] = Some(value);
    }

    /// 获取 `idx` 位置的引用。
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.slots[idx].as_ref()
    }

    /// 获取 `idx` 位置的可变引用。
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        self.slots[idx].as_mut()
    }

    /// 反转查找：返回 `value` 所在的槽位索引。
    #[inline]
    pub fn position_of(&self, value: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.slots
            .iter()
            .position(|slot| slot.as_ref() == Some(value))
    }

    /// 遍历所有槽位 `(idx, Option<&T>)`，包含空槽。
    pub fn iter(&self) -> impl Iterator<Item = (usize, Option<&T>)> {
        self.slots
            .iter()
            .enumerate()
            .map(|(i, slot)| (i, slot.as_ref()))
    }

    /// 迭代所有非空值的拷贝。
    pub fn values_copied(&self) -> impl Iterator<Item = T> + '_
    where
        T: Copy,
    {
        self.slots.iter().filter_map(|slot| *slot)
    }

    /// 迭代所有非空条目 `(idx, &T)`。
    pub fn occupied(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|v| (i, v)))
    }

    /// 迭代所有非空值 `&T`。
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| slot.as_ref())
    }

    /// 迭代所有非空值的可变引用。
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut().filter_map(|slot| slot.as_mut())
    }
}

impl<T> Index<usize> for Table<T> {
    type Output = Option<T>;

    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        &self.slots[idx]
    }
}

impl<T> IndexMut<usize> for Table<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.slots[idx]
    }
}
