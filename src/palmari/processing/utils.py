def get_values_as_in_dict(dict_to_copy: dict, summary_string: str) -> dict:
    print(summary_string)
    dict_to_return = {}
    for key, value in dict_to_copy.items():
        orig_cls = type(value)
        try:
            if orig_cls == bool:
                print("Enter %s (default is %s) : " % (key, value))
                input_v = input(
                    "Press y for true, n for false, anything else will be considered as default : "
                )
                if input_v == "y":
                    input_v = True
                elif input_v == "n":
                    input_v = False
                else:
                    print("Using default %s" % value)
                    input_v = value
            else:
                input_v = input("Enter %s (default is %s) : " % (key, value))
                if len(input_v.strip()) == 0:
                    print("Using default")
                    input_v = value
                else:
                    input_v = orig_cls(input_v)
        except:
            print(
                "Value entered %s is could not be converted to type %s"
                % (value, orig_cls)
            )
            print("Setting %s to default %s" % (key, value))
            input_v = value
        dict_to_return[key] = input_v
    return dict_to_return
