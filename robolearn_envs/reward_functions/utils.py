def convert_reward_cost_fcn(fcn, *args, **kwargs):
    output, output_dict = fcn(*args, **kwargs)

    output *= -1
    output_dict.update((key, value * -1) for key, value in output_dict.items())

    return output, output_dict


