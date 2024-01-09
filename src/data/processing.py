from tensordict import TensorDict


def select_data(data, periods):
    from src.main import variables_time_dependant

    selected_data = {}
    for period_name, period in periods.items():
        data_period = data.sel(time=period)[variables_time_dependant]

        selected_data[period_name] = TensorDict(
            source={k: data_period[k].values for k in data_period.data_vars},
            batch_size=data_period.sizes["time"],
        )

    return selected_data
