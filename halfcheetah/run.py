# Adapted from https://github.com/kinalmehta/Imitation-Learning-Pytorch
# Usage: python run.py --envname HalfCheetah-v2 --render --dagger --M 20 --alpha 0.4

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import tensorflow.compat.v1 as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
tf.disable_v2_behavior()
import numpy as np
import tf_util
import gym
import load_policy
import argparse
from tqdm import tqdm
import random
# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


def env_dims(env):
    return (env.observation_space.shape[0], env.action_space.shape[0])


class TorchModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class ImitateTorch:

    def __init__(self, env):
        input_len, output_len = env_dims(env)
        self.output_len = output_len
        self.model = TorchModel(input_len, output_len)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def train(self, X, Y, epochs=1):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        X, Y = torch.from_numpy(np.float32(X)), torch.from_numpy(np.float32(Y))
        dataset = torch.utils.data.TensorDataset(X, Y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in (range(epochs)):

            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print('[%d] loss: %.6f' % (epoch + 1, running_loss / (i + 1)))

    def forward(self, obs):
        obs_tensor = torch.from_numpy(obs)
        obs_tensor = obs_tensor.to(self.device)
        output = self.model(obs_tensor)
        return np.ndarray.flatten(output.detach().to('cpu').numpy())

    def __call__(self, obs):
        self.model.eval()
        return self.forward(obs)

    def call_with_dropout_randomness(self, obs):
        self.model.train()
        op = self.forward(obs)
        self.model.eval()
        return op

    def save(self, filename):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), filename)
        self.model.to(self.device)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))


class DADaggerPolicy:
    def __init__(self, env, student: ImitateTorch, expert, M, alpha):
        # We have to think something about this. I.e. whether normal dagger is filling this up or not.
        self.CAPACITY = 50000
        self.student = student
        self.expert = expert
        self.fraction_assist = 1.
        self.next_idx = 0
        self.size = 0

        # DADAGGER Variables
        self.M = M
        self.alpha = alpha
        self.save_all = False

        # DADAGGER results
        self.results_file = f"./dadagger_M{M}_alpha{alpha}.txt"
        # This line is written to override the previously saved file if any
        with open(self.results_file, "w") as f:
            f.write("Rewards\n")

        input_len, output_len = env_dims(env)
        self.obs_data = np.empty([self.CAPACITY, input_len])
        self.act_data = np.empty([self.CAPACITY, output_len])

    def __call__(self, obs):
        expert_action = self.expert(obs)
        student_action = self.student(obs)
        self.obs_data[self.next_idx] = np.float32(obs)
        self.act_data[self.next_idx] = np.float32(expert_action)
        self.next_idx = (self.next_idx+1) % self.CAPACITY
        self.size = min(self.size+1, self.CAPACITY)
        if random.random() < self.fraction_assist:
            return (expert_action)
        else:
            return (student_action)

    def expert_data(self):
        return self.obs_data[:self.size], self.act_data[:self.size]

    def save_data(self, observations, actions):
        variances = []
        num_observations = len(observations)

        obs_arr = np.float32(observations)
        student_actions = [
            self.student.call_with_dropout_randomness(obs_arr).reshape(-1, self.student.output_len)
            for i in range(self.M)
        ]
        variances = np.var(np.stack(student_actions, axis=-1), axis=-1).sum(axis=-1)

        index_list = sorted(list(range(num_observations)), key=lambda i: variances[i], reverse=True)
        for t, index in enumerate(index_list):
            # I.e. if save all is true, save all data.
            if (not self.save_all) and (t > self.alpha * num_observations):
                break
            self.obs_data[self.next_idx] = np.float32(observations[index])
            self.act_data[self.next_idx] = np.float32(actions[index])
            self.next_idx = (self.next_idx + 1) % self.CAPACITY
            self.size = min(self.size + 1, self.CAPACITY)

    def save_rewards(self, mean_rewards):
        with open(self.results_file, "a") as f:
            f.write(str(mean_rewards)+","+str(self.size)+"\n")


def get_data(env, policy_fn, num_rollouts, render=False):
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config):
        tf_util.initialize()

        returns = []
        observations = []
        actions = []
        for i in tqdm(range(num_rollouts)):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(np.float32(obs[None, :]))
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()

            if isinstance(policy_fn, DADaggerPolicy):
                policy_fn.save_data(observations, actions)
            returns.append(totalr)

        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        if isinstance(policy_fn, DADaggerPolicy):
            policy_fn.save_rewards(np.mean(returns))

        return np.array(observations), np.array(actions)


def load_data_from_pickle(envname):
    file = open('./expert_data/' + envname + '.pkl', 'rb')
    data = pickle.loads(file.read())
    return data['observations'], data['actions']


def behavior_cloning(env, student, expert, num_rollouts, envname):
    X, Y = get_data(env, expert, num_rollouts)
    student.train(X, Y, epochs=81)
    student.save("./trained_models/" + envname + "_behaviorCloning.pt")


def dagger(env, student, expert, num_rollouts, envname, M, alpha):
    dagger_policy = DADaggerPolicy(env, student, expert, M, alpha)

    for i in tqdm(range(200)):
        if i == 0:
            # Generates the initial dataset
            num_rollouts = num_rollouts
            epochs = 81
            dagger_policy.save_all = True
        else:
            # fraction_assist is set to 0 so that only learner action is considered to collect further data
            num_rollouts = 1
            epochs = 4
            #dagger_policy.save_all = False
            dagger_policy.fraction_assist = 0
        get_data(env, dagger_policy, num_rollouts)
        student.train(dagger_policy.obs_data, dagger_policy.act_data, epochs=epochs)
    student.save("./trained_models/" + envname + "_dagger.pt")


def parse_arguments():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--cloning', action='store_true', default=False)
    group.add_argument('--dagger', action='store_true', default=True)
    parser.add_argument('--envname', type=str, default='HalfCheetah-v2')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=500)
    parser.add_argument('--num_rollouts', type=int, default=50,
                        help='Number of expert roll outs')
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    return (args)


def behavior_cloning_pretrained(student, envname):
    student.load("./trained_models/" + envname + "_behaviorCloning.pt")


def dagger_pretrained(student, envname):
    student.load("./trained_models/" + envname + "_dagger.pt")


def main():
    args = parse_arguments()

    print('loading and building expert policy')
    expert = load_policy.load_policy("./experts/" + args.envname + ".pkl")
    print('loaded and built')

    env = gym.make(args.envname)

    student = ImitateTorch(env)

    if args.use_pretrained:
        if args.cloning:
            behavior_cloning_pretrained(student, args.envname)
        elif args.dagger:
            dagger_pretrained(student, args.envname)

    else:
        if args.cloning:
            behavior_cloning(env, student, expert, args.num_rollouts, args.envname)
        elif args.dagger:
            dagger(env, student, expert, args.num_rollouts, args.envname, args.M, args.alpha)

    if args.render:
        get_data(env, student, 60, args.render)


if __name__ == '__main__':
    main()
