import os

import numpy as np
from matplotlib import pyplot as plt
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from metalearning_benchmarks.benchmarks.quadratic1d_benchmark import Quadratic1D
from metalearning_benchmarks.benchmarks.affine1d_benchmark import Affine1D
from neural_process.neural_process import NeuralProcess
from pprint import pprint


def plot(
    np_model: NeuralProcess,
    benchmark: MetaLearningBenchmark,
    n_task_max: int,
    fig,
    axes,
):
    # determine n_task
    n_task_plot = min(n_task_max, benchmark.n_task)

    # evaluate predictions
    n_samples = 10
    x_min = benchmark.x_bounds[0, 0]
    x_max = benchmark.x_bounds[0, 1]
    x_plt_min = x_min - 0.25 * (x_max - x_min)
    x_plt_max = x_max + 0.25 * (x_max - x_min)
    x_plt = np.linspace(x_plt_min, x_plt_max, 128)
    x_plt = np.reshape(x_plt, (-1, 1))

    # plot predictions
    for l in range(n_task_plot):
        task = benchmark.get_task_by_index(l)
        np_model.adapt(task=task)
        ax = axes[0, l]
        ax.clear()
        ax.scatter(task.x, task.y, marker="x", s=5, color="r")
        for s in range(n_samples):
            mu, _ = np_model.predict(x=x_plt)
            ax.plot(x_plt, mu, color="b", alpha=0.3, label="posterior")
        ax.grid()
        ax.set_title("Predictions")

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def main():
    # logpath
    logpath = os.path.dirname(os.path.abspath(__file__))
    logpath = os.path.join(logpath, "log")
    os.makedirs(logpath, exist_ok=True)

    ## config
    config = dict()
    # model
    config["model"] = "StandardNP"
    # logging
    config["logpath"] = logpath
    # seed
    config["seed"] = 1234
    # meta data
    config["data_noise_std"] = 0.5
    config["n_task_meta"] = 16
    config["n_datapoints_per_task_meta"] = 16
    config["seed_task_meta"] = 1234
    config["seed_x_meta"] = 2234
    config["seed_noise_meta"] = 3234
    # test data
    config["n_task_test"] = 128
    config["n_datapoints_per_task_test"] = 2
    config["seed_task_test"] = 1235
    config["seed_x_test"] = 2235
    config["seed_noise_test"] = 3235

    # generate benchmarks
    benchmark_meta = Affine1D(
        n_task=config["n_task_meta"],
        n_datapoints_per_task=config["n_datapoints_per_task_meta"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_meta"],
        seed_x=config["seed_x_meta"],
        seed_noise=config["seed_noise_meta"],
    )
    benchmark_test = Affine1D(
        n_task=config["n_task_test"],
        n_datapoints_per_task=config["n_datapoints_per_task_test"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_test"],
        seed_x=config["seed_x_test"],
        seed_noise=config["seed_noise_test"],
    )

    # architecture
    config["d_x"] = benchmark_meta.d_x
    config["d_y"] = benchmark_meta.d_y
    config["d_z"] = 2
    config["aggregator_type"] = "BA"
    config["loss_type"] = "MC"
    config["input_mlp_std_y"] = ""
    config["self_attention_type"] = None
    config["f_act"] = "relu"
    config["n_hidden_layers"] = 2
    config["n_hidden_units"] = 16
    config["latent_prior_scale"] = 1.0
    config["decoder_output_scale"] = config["data_noise_std"]

    # training
    config["n_tasks_train"] = int(2 ** 14)
    config["validation_interval"] = int(2 ** 10)
    config["device"] = "cuda"
    config["adam_lr"] = 1e-4
    config["batch_size"] = config["n_task_meta"]
    config["n_samples"] = 16
    config["n_context"] = (
        config["n_datapoints_per_task_test"],
        config["n_datapoints_per_task_test"],
    )

    # generate NP model
    model = NeuralProcess(
        logpath=config["logpath"],
        seed=config["seed"],
        d_x=config["d_x"],
        d_y=config["d_y"],
        d_z=config["d_z"],
        n_context=config["n_context"],
        aggregator_type=config["aggregator_type"],
        loss_type=config["loss_type"],
        input_mlp_std_y=config["input_mlp_std_y"],
        self_attention_type=config["self_attention_type"],
        latent_prior_scale=config["latent_prior_scale"],
        f_act=config["f_act"],
        n_hidden_layers=config["n_hidden_layers"],
        n_hidden_units=config["n_hidden_units"],
        decoder_output_scale=config["decoder_output_scale"],
        device=config["device"],
        adam_lr=config["adam_lr"],
        batch_size=config["batch_size"],
        n_samples=config["n_samples"],
    )
    pprint(model._config)

    # train the model
    n_task_plot = 4
    fig, axes = plt.subplots(
        nrows=1, ncols=n_task_plot, sharex=True, sharey=True, squeeze=False
    )
    callback = lambda np_model: plot(
        np_model=np_model,
        n_task_max=n_task_plot,
        benchmark=benchmark_meta,
        fig=fig,
        axes=axes,
    )
    model.meta_train(
        benchmark_meta=benchmark_meta,
        n_tasks_train=config["n_tasks_train"],
        validation_interval=config["validation_interval"],
        callback=callback,
    )
    plt.show()


if __name__ == "__main__":
    main()
