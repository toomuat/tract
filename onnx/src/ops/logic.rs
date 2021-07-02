use crate::model::OnnxOpRegister;
use crate::model::{ParseResult, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::prelude::tract_itertools::Itertools;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok((Box::new(ops::logic::not()), vec![])));
    reg.insert("And", |_, _| Ok((ops::logic::And.into_hir(), vec![])));
    reg.insert("Or", |_, _| Ok((ops::logic::Or.into_hir(), vec![])));
    reg.insert("Xor", |_, _| Ok((ops::logic::Xor.into_hir(), vec![])));

    reg.insert("Equal", |_, _| Ok((ops::logic::Equals.into_hir(), vec![])));
    reg.insert("Greater", |_, _| Ok((ops::logic::Greater.into_hir(), vec![])));
    reg.insert("Less", |_, _| Ok((ops::logic::Lesser.into_hir(), vec![])));
    reg.insert("LessOrEqual", |_, _| Ok((ops::logic::LesserEqual.into_hir(), vec![])));
    reg.insert("GreaterOrEqual", |_, _| Ok((ops::logic::GreaterEqual.into_hir(), vec![])));

    reg.insert("Where", |_, _| Ok((Box::new(ops::logic::Iff::default()), vec![])));

    reg.insert("If", _if)
}

pub fn _if(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let graph_then = node.get_attr("then_branch")?;
    let graph_else = node.get_attr("else_branch")?;
    let ParseResult { model: mut then_body, unresolved_inputs: unresolved_inputs_then, .. } =
        ctx.parse_graph(graph_then)?;
    let ParseResult { model: mut else_body, unresolved_inputs: unresolved_inputs_else, .. } =
        ctx.parse_graph(graph_else)?;
    let unresolved_inputs = unresolved_inputs_then
        .iter()
        .chain(unresolved_inputs_else.iter())
        .sorted()
        .unique()
        .cloned()
        .collect();
    let then_input_mapping = unresolved_inputs_then
        .iter()
        .map(|i| unresolved_inputs.iter().position(|s| s == i).unwrap() + 1)
        .collect();
    let else_input_mapping = unresolved_inputs_else
        .iter()
        .map(|i| unresolved_inputs.iter().position(|s| s == i).unwrap() + 1)
        .collect();
    Ok((
        Box::new(If { then_body, then_input_mapping, else_body, else_input_mapping }),
        unresolved_inputs,
    ))
}

struct If {
    then_body: InferenceModel,
    then_input_mapping: Vec<usize>,
    else_body: InferenceModel,
    else_input_mapping: Vec<usize>,
}
