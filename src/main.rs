use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{Decoration, Op, StorageClass};
use std::collections::HashMap;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    vertex: String,
    fragment: String,
    output: String,
}

// https://github.com/gfx-rs/rspirv/pull/13
fn assemble_module_into_bytes(module: &rspirv::dr::Module) -> Vec<u8> {
    use rspirv::binary::Assemble;
    use std::mem;
    module
        .assemble()
        .iter()
        .flat_map(|val| (0..mem::size_of::<u32>()).map(move |i| ((val >> (8 * i)) & 0xff) as u8))
        .collect()
}

fn main() {
    let opt = Opt::from_args();

    let vertex = std::fs::read(&opt.vertex).unwrap();
    let fragment = std::fs::read(&opt.fragment).unwrap();

    let vertex = rspirv::dr::load_bytes(&vertex).unwrap();
    let fragment = rspirv::dr::load_bytes(&fragment).unwrap();

    let mut output = fragment.clone();

    let mut output_header = output.header.clone().unwrap();

    let vertex_location_to_id =
        collect_location_to_inst(&vertex, &collect_locations(&vertex), StorageClass::Output);

    let fragment_location_to_id = collect_location_to_inst(
        &fragment,
        &collect_locations(&fragment),
        StorageClass::Input,
    );

    let collected = collect_globals(&vertex);

    for (vertex_location, variable_inst) in &vertex_location_to_id {
        if !fragment_location_to_id.contains_key(vertex_location) {
            debug_assert_eq!(variable_inst.class.opcode, Op::Variable);

            let mut pointer_inst = collected[&variable_inst.result_type.unwrap()].clone();
            let mut type_inst = collected[&pointer_inst.operands[1].unwrap_id_ref()].clone();

            if type_inst.class.opcode == Op::TypeVector {
                let mut scalar_inst = collected[&type_inst.operands[0].unwrap_id_ref()].clone();

                let scalar_result_id = output_header.bound;
                output_header.bound += 1;

                scalar_inst.result_id = Some(scalar_result_id);

                type_inst.operands[0] = Operand::IdRef(scalar_result_id);

                output.types_global_values.push(scalar_inst);
            }

            let type_result_id = output_header.bound;
            output_header.bound += 1;

            type_inst.result_id = Some(type_result_id);

            output.types_global_values.push(type_inst);

            let pointer_result_id = output_header.bound;
            output_header.bound += 1;

            pointer_inst.result_id = Some(pointer_result_id);
            pointer_inst.operands[0] = Operand::StorageClass(StorageClass::Input);
            pointer_inst.operands[1] = Operand::IdRef(type_result_id);

            debug_assert_eq!(pointer_inst.class.opcode, Op::TypePointer);

            output.types_global_values.push(pointer_inst);

            let mut variable_inst = variable_inst.clone();

            let variable_result_id = output_header.bound;
            output_header.bound += 1;

            variable_inst.result_id = Some(variable_result_id);
            variable_inst.result_type = Some(pointer_result_id);
            variable_inst.operands[0] = Operand::StorageClass(StorageClass::Input);

            output.types_global_values.push(variable_inst);

            output.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(variable_result_id),
                    Operand::Decoration(Decoration::Location),
                    Operand::LiteralInt32(*vertex_location),
                ],
            ));

            output.entry_points[0]
                .operands
                .push(Operand::IdRef(variable_result_id));
        }
    }

    output.header = Some(output_header);

    let output = assemble_module_into_bytes(&output);

    std::fs::write(&opt.output, output).unwrap();
}

fn collect_globals(module: &rspirv::dr::Module) -> HashMap<u32, Instruction> {
    let mut globals = HashMap::new();

    for inst in &module.types_global_values {
        if let Some(result_id) = inst.result_id {
            globals.insert(result_id, inst.clone());
        }
    }

    globals
}

fn collect_locations(module: &rspirv::dr::Module) -> HashMap<u32, u32> {
    let mut locations = HashMap::new();

    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate
            && inst.operands[1].unwrap_decoration() == Decoration::Location
        {
            let id = inst.operands[0].unwrap_id_ref();
            let location = inst.operands[2].unwrap_literal_int32();

            locations.insert(id, location);
        }
    }

    locations
}

fn collect_location_to_inst(
    module: &rspirv::dr::Module,
    locations: &HashMap<u32, u32>,
    class: StorageClass,
) -> HashMap<u32, Instruction> {
    let mut location_to_inst = HashMap::new();

    for inst in &module.types_global_values {
        if inst.class.opcode == Op::Variable && inst.operands[0].unwrap_storage_class() == class {
            if let Some(result_id) = inst.result_id {
                if let Some(location) = locations.get(&result_id) {
                    location_to_inst.insert(*location, inst.clone());
                }
            }
        }
    }

    location_to_inst
}