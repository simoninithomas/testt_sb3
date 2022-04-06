import logging
import os

import shutil

from pathlib import Path

from huggingface_hub import HfApi, HfFolder, Repository

import stable_baselines3
from stable_baselines3 import *

import pickle5
import json
import gym
import zipfile
from datetime import datetime


def _generate_config(model, repo_local_path):
    unzipped_model_folder = model

    # Check if the user forgot to mention the extension of the file
    if model.endswith('.zip') is False:
        model += ".zip"

    # Step 1: Unzip the model
    with zipfile.ZipFile(Path(repo_local_path) / model, 'r') as zip_ref:
        zip_ref.extractall(Path(repo_local_path) / unzipped_model_folder)

    # Step 2: Get data (JSON containing infos) and read it
    with open(Path.joinpath(repo_local_path, unzipped_model_folder, 'data')) as json_file:
        data = json.load(json_file)
        # Add system_info elements to our JSON
        data["system_info"] = stable_baselines3.get_system_info(print_info=False)[0]

        # Write our config.json file
    with open(Path(repo_local_path) / 'config.json', 'w') as outfile:
        json.dump(data, outfile)


def _evaluate_agent(model, eval_env, n_eval_episodes, is_deterministic, repo_local_path):
    print("IS DETERMINISTIC", is_deterministic)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes, is_deterministic)

    # Create json evaluation
    evaluate_data = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "is_deterministic": is_deterministic,
        "n_eval_episodes": n_eval_episodes,
        "evaluate_date": datetime.now()
    }

    # Write a JSON file
    with open(Path(repo_local_path) / 'results.json', 'w') as outfile:
        json.dump(evaluate_data, outfile)
    return mean_reward, std_reward

def is_atari(env_id: str) -> bool:
    entry_point = gym.envs.registry.env_specs[env_id].entry_point
    return "AtariEnv" in str(entry_point)


def _generate_replay(model, eval_env, video_length, is_deterministic, repo_local_path):
    env = VecVideoRecorder(
        eval_env,
        "./",  # Temporary video folder
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="",
    )

    obs = env.reset()
    env.reset()
    try:
        print("DO IT")
        for _ in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=is_deterministic)
            obs, _, _, _ = env.step(action)

        # Save the video
        env.close()

        # Move the video

        os.rename(env.video_recorder.path, "replay.mp4")
        shutil.move("replay.mp4", Path(repo_local_path))
    except KeyboardInterrupt:
        pass


def select_tags(env_id):
    """
    if (is_atari(env_id)):
      env = "atari"
    else:
      env = ""
    """

    model_card = f"""
---
  tags:
  - {env_id}
---
"""
    return model_card


def _generate_model_card(model_name, env_id, mean_reward, std_reward):
    model_card = select_tags(env_id)

    model_card += f"""
  # **{model_name}** Agent playing **{env_id}**
  This is a trained model of a **{model_name}** agent playing **{env_id}** using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3).

  ## Evaluation Results
  """

    model_card += f"""
  mean_reward={mean_reward:.2f} +/- {std_reward}
  """

    model_card += """
  ## Usage (with Stable-baselines3)

  TODO: Add your code
  """

    return model_card


def _create_model_card(repo_dir: Path, generated_model_card):
    """Creates a model card for the repository.
    TODO: Add metrics to model-index
    TODO: Use information from common model cards
    """
    readme_path = repo_dir / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
      readme = generated_model_card
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)


def push_to_hub2(model,
                 agent_architecture,
                 env_id,
                 model_name: str,
                 eval_env,

                 is_deterministic,

                 repo_id: str,
                 filename: str,
                 commit_message: str,

                 n_eval_episodes=10,
                 use_auth_token=True,
                 local_repo_path="hub5",
                 video_length=1000,
                 ):
    """
      Upload a model to Hugging Face Hub.
      :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
      :param filename: name of the model zip or mp4 file from the repository
      :param commit_message: commit message
      :param use_auth_token
      :param local_repo_path: local repository path
      """
    huggingface_token = HfFolder.get_token()

    temp = repo_id.split('/')
    organization = temp[0]
    repo_name = temp[1]
    print("REPO NAME: ", repo_name)
    print("ORGANIZATION: ", organization)

    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    repo_url = api.create_repo(
        name=repo_name,
        token=huggingface_token,
        organization=organization,
        private=False,
        exist_ok=True, )

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=use_auth_token)
    repo.git_pull(rebase=True)

    # Step 1: Save the model
    saved_model = model.save(Path(repo_local_path) / model_name)

    # We create two versions of the environment one for video generation and one for evaluation
    replay_env = eval_env

    # Wrap the eval_env around a Monitor
    # eval_env = Monitor(eval_env)
    # replay_env = Monitor(replay_env)

    # Deterministic by default (except for Atari)
    is_deterministic = not is_atari(env_id)
    print("IS DETERMINISTIC", is_deterministic)

    # Step 2: Create a config file
    _generate_config(model_name, repo_local_path)

    # Step 3: Evaluate the agent
    mean_reward, std_reward = _evaluate_agent(model, eval_env, n_eval_episodes, is_deterministic, repo_local_path)

    # Step 4: Generate a video
    _generate_replay(model, replay_env, video_length, is_deterministic, repo_local_path)

    # Step 5: Generate the model card
    generated_model_card = _generate_model_card(agent_architecture, env_id, mean_reward, std_reward)

    _create_model_card(repo_local_path, generated_model_card)

    logging.info(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message=commit_message)

    logging.info(f"View your model in {repo_url}")

    # Todo: I need to have a feedback like:
    # You can see your model here "https://huggingface.co/repo_url"
    print("Your model has been uploaded to the Hub, you can find it here: ", repo_url)
    return repo_url


README_TEMPLATE = """---
tags:
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
---
# TODO: Fill this model card
"""

def _create_model_card(repo_dir: Path):
    """
    Creates a model card for the repository.
    :param repo_dir:
    """
    readme_path = repo_dir / "README.md"
    readme = ""
    if readme_path.exists():
      with readme_path.open("r", encoding="utf8") as f:
          readme = f.read()
    else:
      readme = README_TEMPLATE
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

def _copy_file(filepath: Path, dst_directory: Path):
    """
    Copy the file to the correct directory
    :param filepath: path of the file
    :param dst_directory: destination directory
    """
    dst = dst_directory / filepath.name
    shutil.copy(str(filepath.name), str(dst))


def push_to_hub(repo_id: str,
               filename: str,
               commit_message: str,
               use_auth_token=True,
               local_repo_path="hub"):
    """
      Upload a model to Hugging Face Hub.
      :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
      :param filename: name of the model zip or mp4 file from the repository
      :param commit_message: commit message
      :param use_auth_token
      :param local_repo_path: local repository path
      """
    huggingface_token = HfFolder.get_token()

    temp = repo_id.split('/')
    organization = temp[0]
    repo_name = temp[1]
    print("REPO NAME: ", repo_name)
    print("ORGANIZATION: ", organization)

    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    repo_url = api.create_repo(
        name=repo_name,
        token=huggingface_token,
        organization=organization,
        private=False,
        exist_ok=True, )

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=use_auth_token)
    repo.git_pull(rebase=True)

    # Add the model
    filename_path = os.path.abspath(filename)
    _copy_file(Path(filename_path), repo_local_path)
    _create_model_card(repo_local_path)

    logging.info(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message=commit_message)

    logging.info(f"View your model in {repo_url}")

    # Todo: I need to have a feedback like:
    # You can see your model here "https://huggingface.co/repo_url"
    print("Your model has been uploaded to the Hub, you can find it here: ", repo_url)
    return repo_url



