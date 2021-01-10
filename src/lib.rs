/*
DTC is generic in T: RealNumber.

static fit function (really the constructor) is generic in M: Matrix<T>

 */
use smartcore::math::num::RealNumber;
use std::mem;
use std::ptr;
use smartcore::linalg::{Matrix, BaseVector};

// handle missing values using Option: ? is None, value is Some

fn entropy<T: RealNumber>(count: &Vec<usize>, n: usize) -> T {
    let mut impurity = T::zero();

    for i in 0..count.len() {
        if count[i] > 0 {
            let p = T::from(count[i]).unwrap() / T::from(n).unwrap();
            impurity = impurity - p * p.log2();
        }
    }
    impurity
}

fn information_gain<T: RealNumber>(parent: &Vec<VarInstance<T>>, split: T, num_classes: usize) -> Option<T> {
    let mut total = 0;
    let mut true_child = vec!(0; num_classes);
    let mut false_child = vec!(0; num_classes);
    let mut class_count: Vec<usize> = vec!(0; num_classes);
    let mut true_child_count = 0;
    let mut false_child_count = 0;
    for x in parent.iter() {
        class_count[x.class] += 1;
        total += 1;
        if let Some(n) = x.value {
            if n < split {
                true_child[x.class] += 1;
                true_child_count += 1;
            } else {
                false_child[x.class] += 1;
                false_child_count += 1;
            }
        } else {
            false_child[x.class] += 1;
            false_child_count += 1;
        }
    }

    let ig: f32 = entropy::<f32>(&class_count, total) - (
        (true_child_count as f32 / total as f32) * entropy::<f32>(&true_child, true_child_count) +
        (false_child_count as f32 / total as f32) * entropy::<f32>(&false_child, false_child_count)
    );
    if ig <= 0.0 {None} else {T::from(ig)}
    //let iv: f32 = entropy::<f32>(&vec!(true_child_count, false_child_count), total);
    //println!("ig: {}\tiv: {}", ig, iv);
    //let igr = T::from(ig/iv).unwrap();
    //if igr.is_nan() {None} else {Some(igr)}
}

fn find_splits<T: RealNumber>(values: &Vec<VarInstance<T>>) -> Vec<T> {
    if values.is_empty() {return Vec::new()}
    values.iter()
        .zip(values[1..].iter())
        .filter(|(x, y)| x.class != y.class)
        .filter_map(|(x, y)| {
            match (x.value, y.value) {
                (Some(n), Some(m)) => Some((n+m)/T::from(2.0).unwrap()),
                (Some(n), None) => Some(n),
                (None, Some(n)) => Some(n),
                _ => None,
            }
        }).collect()
}

fn best_split<T: RealNumber>(values: &Vec<VarInstance<T>>, num_classes: usize) -> Option<(T, T)> {
    find_splits(values).iter()
        .map(|split| (*split, information_gain(values, *split, num_classes)))
        .filter_map(|(val, score)| if score.is_some() {Some((val, score.unwrap()))} else {None})
        .max_by(|(_, score1), (_, score2)| score1.partial_cmp(score2).unwrap())
}

#[derive(Copy, Clone, Debug)]
struct VarInstance<T: RealNumber> {
    value: Option<T>,
    class: usize,
}

#[derive(Clone)]
struct SampleInstance<T: RealNumber> {
    value: Vec<Option<T>>,
    class: usize,
}

#[derive(Clone)]
struct NodeRecord<T: RealNumber> {
    samples: Vec<SampleInstance<T>>,
    record: Vec<Vec<VarInstance<T>>>,
    num_samples: usize,
    num_classes: usize,
    class: usize,
}

impl<T: RealNumber> NodeRecord<T> {
    /// Add example to this record. This operation preserves the sorted quality of the record.
    /// Returns false if example still matches class of record, true otherwise
    fn update(&mut self, example: &Vec<Option<T>>, class: usize) -> bool {
        self.samples.push(SampleInstance { value: example.clone(), class });
        for i in 0..example.len() {
            self.record[i].push(VarInstance { value: example[i], class } );
            let mut index = self.num_samples;
            if index == 0 {continue}
            while self.record[i][index].value < self.record[i][index-1].value {
                self.record[i].swap(index, index - 1);
                index -= 1;
                if index == 0 {break}
            }
        }
        if self.num_samples == 0 {
            self.class = class;
        }
        self.num_samples += 1;
        class != self.class
    }

    fn merge(&mut self, other: &NodeRecord<T>) {
        for sample in &other.samples {
            self.samples.push(sample.clone());
        }
        let mut new_record = vec!(Vec::with_capacity(
            self.num_samples + other.num_samples); self.record.len());
        for i in 0..self.record.len() {
            let mut this_index: usize = 0;
            let mut that_index: usize = 0;
            while this_index < self.num_samples && that_index < other.num_samples {
                if self.record[i][this_index].value < other.record[i][that_index].value {
                    new_record[i].push(self.record[i][this_index]);
                    this_index += 1;
                } else {
                    new_record[i].push(other.record[i][that_index]);
                    that_index += 1;
                }
            }
            if this_index == self.num_samples {
                while that_index < other.num_samples {
                    new_record[i].push(other.record[i][that_index]);
                    that_index += 1;
                }
            } else if that_index == other.num_samples {
                while this_index < self.num_samples {
                    new_record[i].push(self.record[i][this_index]);
                    this_index += 1;
                }
            }
        }
        self.record = new_record;
        self.num_samples += other.num_samples;
        self.update_class();
    }

    // lists of VarInstance are sorted, hence we cannot get the original samples back from them.
    fn get_examples(&self) -> (Vec<Vec<Option<T>>>, Vec<usize>) {
        let mut classes = Vec::new();
        let samples: Vec<_> = self.samples.iter()
            .map(|s| {
                classes.push(s.class);
                s.value.clone()
            }).collect();
        (samples, classes)
    }

    fn clear(&mut self) {
        self.samples.clear();
        for attr in &mut self.record {
            attr.clear();
        }
        self.num_samples = 0;
    }

    fn find_best_test(&self) -> Option<(T, T, usize)> {
        if self.num_samples == 0 { return None; }
        let first_class = self.record[0][0].class;
        let mut uniform = true;
        for var in &self.record[0] {
            if var.class != first_class {
                uniform = false;
                break;
            }
        }
        if uniform { return None; }
        let mut cur_max: Option<(T, T, usize)> = None;
        for attr in 0..self.record.len() {
            if let Some((split, score)) = best_split(&self.record[attr], self.num_classes) {
                if let Some(max) = cur_max {
                    cur_max = if score > max.1 {Some((split, score, attr))} else {cur_max};
                } else {
                    cur_max = Some((split, score, attr));
                }
            }
        }
        cur_max
    }

    fn update_class(&mut self) {
        let mut class_count = vec!(0usize; self.num_classes);
        for var in &self.samples {
            class_count[var.class] += 1;
        }
        let max = class_count.iter()
            .enumerate()
            .max_by(|(_, val1), (_, val2)| val1.cmp(val2))
            .unwrap().0;
        self.class = max;
    }

    fn classes(&self) -> Vec<usize> {
        let mut class_count = vec!(0usize; self.num_classes);
        for var in &self.samples {
            class_count[var.class] += 1;
        }
        class_count
    }
}

struct Decision<T: RealNumber> {
    split_feature: usize,
    split_value: T,
    true_child: Box<Node<T>>,
    false_child: Box<Node<T>>,
}

impl<T: RealNumber> Decision<T> {
    fn test(&self, sample: &Vec<Option<T>>) -> bool {
        if let Some(n) = sample[self.split_feature] {
            n < self.split_value
        } else {
            false
        }
    }
}

pub struct Node<T: RealNumber> {
    record: NodeRecord<T>,
    decision: Option<Decision<T>>,
    stale: bool,
}

impl<T: RealNumber> Node<T> {
    pub fn show(&mut self, indent: i32) {
        for _ in 0..indent {
            print!("\t");
        }
        if let Some(decision) = &mut self.decision {
            print!("Decision: attr{} < {}", decision.split_feature, decision.split_value);
            print!(" class: {} - {:?}", self.record.class, self.record.classes());
            println!(" samples: {}", self.record.num_samples);
            decision.true_child.show(indent + 1);
            decision.false_child.show(indent + 1);
        } else {
            print!("Class: {} - {:?}", self.record.class, self.record.classes());
            println!(" samples: {}", self.record.num_samples);
        }
    }

    fn new(num_attributes: usize, num_classes: usize) -> Node<T> {
        let record = NodeRecord {
            samples: Vec::new(),
            record: vec!(Vec::new(); num_attributes),
            num_samples: 0,
            num_classes,
            class: 0,
        };
        Node {
            record,
            decision: None,
            stale: false,
        }
    }

    pub fn incremental_update(&mut self, sample: &Vec<Option<T>>, class: usize) {
        self.add_example(sample, class);
        self.ensure_best_test();
    }

    pub fn fit<M: Matrix<T>>(values: &M, classes: &M::RowVector) -> Node<T> {
        let mut classes_vec: Vec<_> = classes.to_vec().iter().map(|x| x.to_u32().unwrap()).collect();
        classes_vec.sort();
        classes_vec.dedup();
        let num_classes = classes_vec.len();
        let mut root = Node::new(values.shape().1, num_classes);
        for i in 0..values.shape().0 {
            let sample: Vec<_> = values.get_row_as_vec(i).iter()
                .map(|x| Some(*x))
                .collect();
            root.add_example(&sample, classes.get(i).to_usize().unwrap());
        }
        //root.ensure_best_test();
        root
    }

    pub fn predict_all<M: Matrix<T>>(&self, values: &M) -> M::RowVector {
        let mut prediction = M::ones(1, values.shape().0);
        for i in 0..values.shape().0 {
            let sample: Vec<Option<T>> = values.get_row_as_vec(i).iter().map(|x| Some(*x)).collect();
            prediction.set(0, i, T::from(self.predict(&sample)).unwrap());
        }
        prediction.to_row_vector()
    }

    fn add_all_examples(&mut self, values: &Vec<Vec<Option<T>>>, classes: &Vec<usize>){
        for i in 0..classes.len() {
            self.add_example(&values[i], classes[i]);
        }
    }

    fn predict(&self, sample: &Vec<Option<T>>) -> usize {
        if let Some(decision) = &self.decision {
            if decision.test(sample) {
                decision.true_child.predict(sample)
            } else {
                decision.false_child.predict(sample)
            }
        } else {
            self.record.class
        }
    }

    fn add_example(&mut self, example: &Vec<Option<T>>, class: usize) {
        if let Some(decision) = &mut self.decision {
            self.record.update(example, class);
            if decision.test(example) {
                decision.true_child.add_example(example, class);
            } else {
                decision.false_child.add_example(example, class);
            }
            self.stale = true;
        } else {
            if self.record.update(example, class) {
                if let Some((split_value, _, split_feature)) = self.record.find_best_test() {
                    let n = self.record.record.len();
                    let decision = Decision {
                        split_feature,
                        split_value,
                        true_child: Box::new(Node::new(n, self.record.num_classes)),
                        false_child: Box::new(Node::new(n, self.record.num_classes)),
                    };
                    self.decision = Some(decision);
                    self.stale = false;
                }
                if let Some(decision) = &mut self.decision {
                    let (samples, classes) = self.record.get_examples();
                    for i in 0..self.record.num_samples {
                        if decision.test(&samples[i]) {
                            decision.true_child.add_example(&samples[i], classes[i]);
                        } else {
                            decision.false_child.add_example(&samples[i], classes[i]);
                        }
                    }
                } else { // case: making decision node failed
                    self.record.update_class();
                }
            }
        }
    }

    fn transpose_tree(&mut self, test_split: T, test_feature: usize) {
        let mut left_transpose = false;
        let mut right_transpose = false;
        let mut left_leaf = false;
        let mut right_leaf = false;
        if let Some(decision) = &mut self.decision {
            match (&decision.true_child.decision, &decision.false_child.decision) {
                (Some(d1), Some(d2)) => {
                    left_transpose = d1.split_value != test_split || d1.split_feature != test_feature;
                    right_transpose = d2.split_value != test_split || d2.split_feature != test_feature;
                }
                (Some(d1), None) => {
                    left_transpose = d1.split_value != test_split || d1.split_feature != test_feature;
                    right_leaf = true;
                }
                (None, Some(d1)) => {
                    right_transpose = d1.split_value != test_split || d1.split_feature != test_feature;
                    left_leaf = true;
                }
                (None, None) => {
                    left_leaf = true;
                    right_leaf = true;
                }
            }
            if left_transpose { decision.true_child.transpose_tree(test_split, test_feature); }
            if right_transpose { decision.false_child.transpose_tree(test_split, test_feature); }
            match (left_leaf, right_leaf) {
                (false, false) => {
                    if let (Some(d1), Some(d2)) =
                            (&mut decision.true_child.decision, &mut decision.false_child.decision) {
                        mem::swap(&mut d1.false_child, &mut d2.true_child);

                        decision.true_child.record.clear();
                        decision.true_child.record.merge(&d1.true_child.record);
                        decision.true_child.record.merge(&d1.false_child.record);
                        decision.true_child.stale = true;
                        d1.split_feature = decision.split_feature;
                        d1.split_value = decision.split_value;

                        decision.false_child.record.clear();
                        decision.false_child.record.merge(&d2.true_child.record);
                        decision.false_child.record.merge(&d2.false_child.record);
                        decision.false_child.stale = true;
                        d2.split_feature = decision.split_feature;
                        d2.split_value = decision.split_value;

                        decision.split_value = test_split;
                        decision.split_feature = test_feature;
                    }
                }
                (true, false) => unsafe {
                    let (samples, classes) = decision.true_child.record.get_examples();
                    let d1 = decision.false_child.decision.as_mut().unwrap() as *mut Decision<T>;
                    self.record.clear();
                    self.record.merge(&decision.false_child.record);
                    ptr::swap(d1, decision as *mut Decision<T>);
                    self.add_all_examples(&samples, &classes);
                }
                (false, true) => unsafe {
                    let (samples, classes) = decision.false_child.record.get_examples();
                    let d1 = decision.true_child.decision.as_mut().unwrap() as *mut Decision<T>;
                    self.record.clear();
                    self.record.merge(&decision.true_child.record);
                    ptr::swap(d1, decision as *mut Decision<T>);
                    self.add_all_examples(&samples, &classes);
                }
                (true, true) => {
                    self.record.clear();
                    decision.split_value = test_split;
                    decision.split_feature = test_feature;
                    let (l_samples, l_classes) = decision.true_child.record.get_examples();
                    let (r_samples, r_classes) = decision.false_child.record.get_examples();
                    decision.true_child.record.clear();
                    decision.false_child.record.clear();
                    self.add_all_examples(&l_samples, &l_classes);
                    self.add_all_examples(&r_samples, &r_classes);
                }
            }
        }
    }

    pub fn ensure_best_test(&mut self) {
        if self.stale {
            if let Some((split_value, split_score, split_feature)) = self.record.find_best_test() {
                let mut transpose = false;
                if let Some(decision) = &mut self.decision {
                    let cur_score = information_gain(&self.record.record[decision.split_feature],
                                                     decision.split_value, self.record.num_classes);
                    if let Some(score) = cur_score {
                        if score >= split_score {
                            self.stale = false;
                            return
                        }
                    }
                    transpose = split_value != decision.split_value || split_feature != decision.split_feature;
                }
                if transpose {
                    self.transpose_tree(split_value, split_feature);
                }
                if let Some(decision) = &mut self.decision {
                    self.stale = false;
                    decision.true_child.ensure_best_test();
                    decision.false_child.ensure_best_test();
                }
            } else {
                self.decision = None;
                self.stale = false;
            }
        }
        self.record.update_class();
    }
}

#[cfg(test)]
mod tests {
    use crate::entropy;

    #[test]
    fn entropy_test() {
        let h: f32 = entropy(&vec!(1,1), 2);
        assert_eq!(2 + 2, 4);
    }
}
