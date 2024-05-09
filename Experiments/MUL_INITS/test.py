import math
import wandb

N_STEPS = 100

wandb.init(project="Optim_Alg", name="debug")
table = wandb.Table(columns=["metric", "value", "step"])
for epoch in range(N_STEPS):
    log = {}
    log['main/metric'] = epoch / N_STEPS  # some main metric

    # some other metrics I want to have all on 1 plot
    other_metrics = {}
    for j in range(10):
        log[f'other_metrics/metric_{j}'] = math.sin(j * math.pi * (epoch / N_STEPS))
        table.add_data(f'other_metrics/metric_{j}', log[f'other_metrics/metric_{j}'], wandb.run.step)
    wandb.log(log)

wandb.log({"multiline": wandb.plot_table(
    "wandb/line/v0", table, {"x": "step", "y": "value", "groupKeys": "metric"}, {"title": "Multiline Plot"})
})

wandb.finish()