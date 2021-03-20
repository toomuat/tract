use tract_hir::internal::*;

use crate::model::{ParsingContext, TfOpRegister};
use crate::tfpb::tensorflow::NodeDef;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("ResizeBilinear", resize_bilinear);
}

fn resize_bilinear(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let align_corners: i32 = pb.get_attr_opt_int("align_corners")?.unwrap_or(0);
    Ok(expand(Resize::new(align_corners == 1)))
}

#[derive(Clone, new, Debug, Hash)]
pub struct Resize {
    align_corners: bool,
}

impl_dyn_hash!(Resize);

impl Expansion for Resize {
    fn name(&self) -> Cow<str> {
        "Resize".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_input_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[1].shape[0], 2.to_dim())?;
        s.equals(&outputs[0].rank, 4)?;
        s.given(&inputs[1].value, move |s, shape| {
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            s.equals(&outputs[0].shape[1], &shape[0])?;
            s.equals(&outputs[0].shape[2], &shape[1])?;
            Ok(())
        })?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[0].shape[3], &inputs[0].shape[3])?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::ops::resize::CoordTransformer::*;
        use tract_hir::ops::resize::Interpolator::*;
        use tract_hir::ops::resize::Nearest::*;
        let coord_transformer = if self.align_corners { AlignCorners } else { HalfPixel };
        let op = tract_hir::ops::resize::Resize {
            coord_transformer,
            interpolator: Linear,
            nearest: Floor,
            optional_sizes_input: Some(1),
            optional_scales_input: None,
        };
        let input_fact = model.outlet_fact(inputs[0])?.clone();
        use tract_core::ops::array::ConcatSlice::*;
        use tract_core::ops::array::TypedConcat;
        let shape = model.wire_node(
            format!("{}.to_dim", prefix),
            tract_core::ops::cast::cast(TDim::datum_type()),
            &[inputs[1]],
        )?;
        let shape = model.wire_node(
            format!("{}.shape", prefix),
            TypedConcat::new(
                0,
                tvec![
                    Const(rctensor1(&[input_fact.shape[0].clone()])),
                    Var,
                    Const(rctensor1(&[input_fact.shape[3].clone()])),
                ],
            ),
            &shape,
        )?[0];
        model.wire_node(prefix, op, &[inputs[0], shape])
    }

    op_tf!();
}
