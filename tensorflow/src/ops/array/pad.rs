use crate::tfpb::tensorflow::NodeDef;
use tract_hir::{internal::*, tract_core::itertools::Itertools};

use crate::model::ParsingContext;

#[derive(Debug, Clone, Default, Hash)]
pub struct Pad;

impl_dyn_hash!(Pad);

pub fn pad(_ctx: &ParsingContext, _pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    Ok(expand(Pad))
}

impl Expansion for Pad {
    fn name(&self) -> Cow<str> {
        "Pad".into()
    }

    op_tf!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let input = &inputs[0];
        let padding = &inputs[1];
        let output = &outputs[0];
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&output.datum_type, &input.datum_type)?;
        s.equals(&input.rank, &output.rank)?;
        s.equals(&padding.rank, 2)?;
        s.equals(&padding.shape[0], input.rank.bex().to_dim())?;
        s.equals(&padding.shape[1], 2.to_dim())?;
        s.given(&input.rank, move |s, rank| {
            for d in 0..rank as usize {
                s.equals(
                    &output.shape[d],
                    input.shape[d].bex()
                        + padding.value[d][0].bex().to_dim()
                        + padding.value[d][1].bex().to_dim(),
                )?
            }
            Ok(())
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let pads =
            model.outlet_fact(inputs[1])?.konst.as_ref().context("Expect pads to be constant")?;
        let pads = pads.cast_to::<u32>()?;
        let pads = pads.as_slice::<u32>()?;
        let op = tract_core::ops::array::Pad::new(
            (0..pads.len() / 2).map(|d| (pads[2 * d] as usize, pads[2 * d + 1] as usize)).collect(),
            tract_core::ops::array::PadMode::Constant(
                Tensor::zero_scalar_dt(model.outlet_fact(inputs[0])?.datum_type)?.into_arc_tensor(),
            ),
        );
        model.wire_node(prefix, op, &[inputs[0]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_0() {
        let inputs = tvec![rctensor2(&[[1, 2, 3], [4, 5, 6]]), rctensor2(&[[1, 1], [2, 2]]),];

        let expected: TVec<_> = tvec!(rctensor2(&[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0, 0],
            [0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]));

        assert_eq!(expand(Pad).eval(inputs).unwrap(), expected);
    }
}
