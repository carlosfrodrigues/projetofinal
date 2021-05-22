use rustler::{Encoder, Env, Error, Term};
//use ord_subset::OrdSubsetIterExt;
use std::collections::HashMap;

#[derive(rustler::NifStruct)]
#[module = "Knn.Point"]
struct Point {
    fields: Vec<f64>,
    class: i8
}

struct Distance {
    value: f64,
    class: i8
}

mod atoms {
    rustler::rustler_atoms! {
        atom ok;
        atom error;
        atom __true__ = "true";
        atom __false__ = "false";
    }
}

rustler::rustler_export_nifs! {
    "Elixir.Knn",
    [
        ("predict", 2, predict)
    ],
    None
}

fn euclidean(x: &Point, y: &[f64]) -> Distance {
    let distance =
    x.fields.as_slice().iter()
        .zip(y.iter())
        .fold(0.0, |s, (&a, &b)| s + (a - b) * (a - b));
    Distance {value: distance, class: x.class}
}

fn classify(training: &[Point], fields: &[f64], k: i8) -> i8 {
    let mut distance = training.iter()
    .map(|p| euclidean(p, fields)).collect::<Vec<Distance>>();

    distance.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
    let indexes = distance[0..k as usize].iter().map(|d| d.class).collect::<Vec<i8>>();
    mode(indexes.as_slice()).unwrap()
}

fn mode(numbers: &[i8]) -> Option<i8> {
    let mut counts = HashMap::new();

    numbers.iter().copied().max_by_key(|&n| {
        let count = counts.entry(n).or_insert(0);
        *count += 1;
        *count
    })
}

fn predict<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let training_set: Vec<Point> = args[0].decode()?;
    let mut validation_sample: Vec<Point> = args[1].decode()?;
    validation_sample.truncate(1);
    let num_correct = validation_sample.iter()
    .filter(|x| {
        classify(training_set.as_slice(), x.fields.as_slice(), 5) == x.class
    })
    .count();
    //println!("Porcentagem de acertos: {}%",
    //         num_correct as f64 / validation_sample.len() as f64 * 100.0);

    Ok((atoms::ok(), num_correct).encode(env))
}
