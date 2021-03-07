use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;
use tract_hir::ops::nn;

pub fn argmax(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    dbg!(pb);
    todo!();
}

#[derive(Clone, Debug, Hash, PartialEq)]
struct ArgMax;

impl_dyn_hash!(ArgMax);

impl Expansion for ArgMax {
    fn name(&self) -> Cow<str> {
        "ArgMax".into()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dim = model
            .outlet_fact(inputs[1])?
            .konst
            .as_ref()
            .context("Dim is expected to be a constant")?
            .cast_to_scalar::<i64>()?;
        let op = nn::Reduce::new(Some(vec![dim]), false, nn::Reducer::ArgMax(false));
        op.wire(prefix, model, inputs)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&inputs, 1)?;
        s.equals(&outputs[0].datum_type, i64::datum_type())?;
        s.equals(&outputs[0].rank, inputs[0].rank.bex() - 1)?;
        s.given_2(&inputs[0].rank, &inputs[1].value, move |s, rank, dim| {
            let dim = dim.cast_to_scalar::<i64>()? as usize;
            for ix in 0..rank as usize {
                if ix < dim {
                    s.equals(&inputs[0].shape[ix], &outputs[0].shape[ix])?;
                } else if ix > dim {
                    s.equals(&inputs[0].shape[ix], &outputs[0].shape[ix - 1])?;
                };
            }
            Ok(())
        })?;
        Ok(())
    }

    op_tf!();
}
