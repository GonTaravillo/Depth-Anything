import onnx

def validate_model(onnx_path):
    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)
    
    print("Checking model...")
    onnx.checker.check_model(model)
    print("Model is valid!")
    
    print("\nModel Inputs:")
    for input in model.graph.input:
        print(input.name, end=': ')
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither
                if (d.HasField("dim_value")):
                    print(d.dim_value, end=', ')
                elif (d.HasField("dim_param")):
                    print(d.dim_param, end=', ')
                else:
                    print('?', end=', ')
        else:
            print("unknown shape", end="")
        print()
        
    print("\nModel Outputs:")
    for output in model.graph.output:
        print(output.name, end=': ')
        tensor_type = output.type.tensor_type
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    print(d.dim_value, end=', ')
                elif (d.HasField("dim_param")):
                    print(d.dim_param, end=', ')
                else:
                    print('?', end=', ')
        else:
            print("unknown shape", end="")
        print()

if __name__ == "__main__":
    validate_model("../depth_anything_nano_epoch10.onnx")
