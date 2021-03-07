use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

use tract_hir::ops::resize::*;

pub fn resize(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let coord_transformer =
        match node.get_attr_opt("coordinate_transformation_mode")?.unwrap_or("half_pixel") {
            "align_corners" => CoordTransformer::AlignCorners,
            "half_pixel" => CoordTransformer::HalfPixel,
            s => todo!("coordinate_transformation_mode: {}", s),
        };
    let interpolator = match node.get_attr("mode")? {
        "linear" => Interpolator::Linear,
        s => todo!("mode: {}", s),
    };
    let nearest = match node.get_attr_opt("nearest_mode")?.unwrap_or("round_prefer_floor") {
        "floor" => Nearest::Floor,
        "round_prefer_floor" => Nearest::RoundPreferFloor,
        s => todo!("nearest_mode: {}", s),
    };
    let mut options = crate::model::optional_inputs(node).skip(2);
    Ok((
        Box::new(Resize {
            optional_scales_input: options.next().unwrap(),
            optional_sizes_input: options.next().unwrap(),
            coord_transformer,
            interpolator,
            nearest,
        }),
        vec![],
    ))
}
