import matplotlib.pyplot as plt

if __name__ == "__main__":
    params = [
        (1, 1.0),
        (10, 0.1),
        (10, 0.2),
        (10, 0.4),
        (25, 0.1),
        (25, 0.2),
        (25, 0.4),
    ]

    fig, ax = plt.subplots(1, 1, dpi=200)

    expert_reward = 0
    for M, alpha in params:
        rewards = []
        filename = f"./results/archive/dadagger_M{M}_alpha{alpha}.txt"
        with open(filename, "r") as f:
            f.readline()
            expert_reward += float(f.readline().split(",")[0])
            while True:
                line = f.readline()
                if not line:
                    break
                rewards.append(float(line.split(",")[0]))
        ax.plot(rewards, label=f"M={M}, alpha={alpha}")
    expert_reward /= len(params)

    ax.hlines(
        y=expert_reward,
        xmin=0,
        xmax=len(rewards),
        label="Expert return",
        colors=["red"],
    )
    ax.set_xlabel("DADAgger iteration")
    ax.set_ylabel("Mean return")
    ax.set_ylim(1800, None)
    fig.legend(loc="right")
    fig.tight_layout()
    plt.show()
