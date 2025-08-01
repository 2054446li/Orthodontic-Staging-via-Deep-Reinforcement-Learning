import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import json
from utils import utils_np
import open3d as o3d
from env.ortho_env_copy import OrthoEnv
from tqdm import tqdm
from utils.prioritized_memory_copy import Memory
from utils.running_mean_std_copy import Scale,RunningMeanStdState
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

###################
error_cases=['C01002722632.json', 'C01002722812.json', 'C01002724937.json', 'C01002726883.json', 'C01002728672.json', 'C01002737908.json', 'C01002739797.json', 'C01002739809.json', 'C01002740294.json', 'C01002742285.json', 'C01002742814.json', 'C01002743376.json', 'C01002748270.json', 'C01002752736.json', 'C01002753894.json', 'C01002757078.json', 'C01002760218.json', 'C01002760285.json', 'C01002762513.json', 'C01002764234.json', 'C01002770466.json', 'C01002772985.json', 'C01002774123.json', 'C01002774594.json', 'C01002775269.json', 'C01002784742.json', 'C01002791706.json', 'C01002792886.json', 'C01002796891.json', 'C01002800505.json', 'C01002807805.json', 'C01002809896.json', 'C01002810292.json', 'C01002811406.json', 'C01002811855.json', 'C01002812430.json', 'C01002817413.json', 'C01002818931.json', 'C01002828437.json', 'C01002828482.json', 'C01002837246.json', 'C01002838124.json', 'C01002838337.json', 'C01002840587.json', 'C01002844772.json', 'C01002849621.json', 'C01002722823.json', 'C01002725118.json', 'C01002725736.json', 'C01002727154.json', 'C01002727817.json', 'C01002728762.json', 'C01002735973.json', 'C01002736749.json', 'C01002736806.json', 'C01002737627.json', 'C01002738954.json', 'C01002742982.json', 'C01002744298.json', 'C01002744513.json', 'C01002744715.json', 'C01002745255.json', 'C01002746492.json', 'C01002746762.json', 'C01002746784.json', 'C01002747392.json', 'C01002748258.json', 'C01002750688.json', 'C01002751746.json', 'C01002752343.json', 'C01002752398.json', 'C01002756167.json', 'C01002761703.json', 'C01002763288.json', 'C01002763514.json', 'C01002764458.json', 'C01002764650.json', 'C01002767170.json', 'C01002767967.json', 'C01002770411.json', 'C01002772389.json', 'C01002772402.json', 'C01002772660.json', 'C01002775270.json', 'C01002776709.json', 'C01002778059.json', 'C01002778116.json', 'C01002781299.json', 'C01002782256.json', 'C01002782469.json', 'C01002785002.json', 'C01002787969.json', 'C01002788634.json', 'C01002791605.json', 'C01002791650.json', 'C01002792909.json', 'C01002793517.json', 'C01002795801.json', 'C01002796969.json', 'C01002799164.json', 'C01002800279.json', 'C01002801236.json', 'C01002805533.json', 'C01002808367.json', 'C01002811237.json', 'C01002811934.json', 'C01002821159.json', 'C01002823780.json', 'C01002824747.json', 'C01002830711.json', 'C01002831149.json', 'C01002834276.json', 'C01002835435.json', 'C01002836043.json', 'C01002836706.json', 'C01002840767.json', 'C01002844996.json', 'C01002846987.json', 'C01002847045.json', 'C01002722788.json', 'C01002747516.json', 'C01002774628.json', 'C01002785507.json', 'C01002796914.json', 'C01002803328.json', 'C01002815310.json', 'C01002735210.json', 'C01002737403.json', 'C01002756831.json', 'C01002763198.json', 'C01002763390.json', 'C01002775708.json', 'C01002789185.json', 'C01002801630.json', 'C01002814870.json', 'C01002826165.json', 'C01002725466.json', 'C01002726265.json', 'C01002745749.json', 'C01002757180.json', 'C01002766258.json', 'C01002767675.json', 'C01002771849.json', 'C01002780423.json', 'C01002801146.json', 'C01002847483.json', 'C01002744748.json', 'C01002776833.json', 'C01002790266.json', 'C01002796879.json', 'C01002826705.json', 'C01002827975.json', 'C01002838720.json', 'C01002845920.json']

up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(ids)}  
#####################

diff=np.load("data/clean_diff.npy").astype(np.float32)
teeth_ids=ids
teeth_ids_pos=[oid[id] for id in teeth_ids]
max_diff=diff[teeth_ids_pos,:]
max_diff_tensor=torch.from_numpy(max_diff)
shape_dir="data/feature_pointr_108d"
hull_dir="data/hull_512" 
teeth_mean=np.load("data/mean.npy")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 2024
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.90
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    policy_frequency: int = 50
    plot_frequency: int = 5000
    """the frequency of training policy (delayed)"""
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    train_dir='/home/ly/OrthodonticStaging/RL-Staging/data/train_data'
    test_dir='/home/ly/OrthodonticStaging/RL-Staging/data/tmp_data'
    n_head=4
    n_layers=2
    dropout=0.1
    hidden_dim=256
    state_dim=144
    q_learning_rate=1e-4
    actor_learning_rate=1e-5
    n_step=3
    q_alpha=0.5
    collision_punishment=10
    alpha_trans=100
    alpha_rotation=100
    q_weight_decay=1e-2
    actor_weight_decay=1e-4
    thred_completion=0.3
    # save_path="result_L1_part3/28_collpun_10_lr_0.0001_gamma_0.9_noise_0.1_256_tem_0.5_L1_100_L2_0_relu/1899981.cleanrl_model"
    save_path=None
    lambda_mask=100
    alpha_L2=0
    temperature = 0.1
    anneal_lr=False
    expert_weight=10
    weight_sum=6
    alpha_smooth=10


def weight_initialization(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for weights
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  # Initialize biases to zero

# 自定义权重初始化函数
import torch.nn.init as init
def weights_init(m):
    if isinstance(m, nn.Linear):
        # 使用He初始化（适用于ReLU激活函数）
        nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)  # 偏置初始化为0
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)  # LayerNorm权重初始化为1
        init.zeros_(m.bias)  # LayerNorm偏置初始化为0
    elif isinstance(m, nn.Embedding):
        init.xavier_normal_(m.weight)  # 嵌入层使用Xavier初始化



class QNetwork(nn.Module):
    def __init__(self, n_teeth=28, state_dim=18, action_dim=9, shape_dim=108,
                 n_heads=4, n_layers=2, hidden_dim=64, dropout=0.1):
        super(QNetwork, self).__init__()
        self.n_teeth = n_teeth
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # embedding 输入维度 = state + action
        input_dim = state_dim + action_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.shape_embedding = nn.Linear(shape_dim, shape_dim)
        self.shape_ln = nn.LayerNorm(shape_dim)
        self.relu=nn.ReLU()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout),
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.pos_embedding = nn.Embedding(n_teeth, hidden_dim)  
        self.fc2=nn.Linear(self.n_teeth,1) 
        self.apply(weights_init)

    def forward(self, state, action):
        """
        state: (batch_size, n_teeth, state_dim)
        action: (batch_size, n_teeth, action_dim)
        """
        shape = self.shape_embedding(state[:, :, 36:])
        shape = self.shape_ln(shape)
        shape=self.relu(shape)
        
        state_features = state[:, :, :36]
        
        x = torch.cat([state_features, shape, action], dim=-1)  # (batch, n_teeth, input_dim)
        # x=torch.cat([state_features,action],dim=-1)
        x = self.embedding(x)  # (batch, n_teeth, hidden_dim)
        
        # 添加 2D positional embedding
        pos_embed=self.pos_embedding(torch.tensor(list(range(self.n_teeth)),device=device))
        # (n_teeth, hidden_dim)
        x = x + pos_embed.unsqueeze(0)  # broadcast over batch
        
        # Transformer 输入 (seq_len, batch, hidden_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)  # (batch, n_teeth, hidden_dim)
        
        q_per_tooth = self.fc_out(x).squeeze(-1)  # (batch, n_teeth)

        q_value=self.fc2(q_per_tooth)
        # q_value = q_per_tooth.mean(dim=1, keepdim=True)  # (batch, 1)
        
        return q_value


class Actor(nn.Module):
    def __init__(self, n_teeth=28, state_dim=135, action_dim=9, shape_dim=108,
                 n_heads=4, n_layers=2, hidden_dim=64, dropout=0.1, action_scale=1.0):
        super(Actor, self).__init__()
        self.n_teeth = n_teeth
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        input_dim = state_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.shape_embedding = nn.Linear(shape_dim, shape_dim)
        self.shape_ln = nn.LayerNorm(shape_dim)
        self.relu=nn.ReLU()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout),
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 将输出限制在 [-1,1]
        )
        
        self.weight_out=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2,1),
        )
        self.sigmod=nn.Sigmoid()

        self.pos_embedding = nn.Embedding(n_teeth, hidden_dim)
        
        self.apply(weights_init)
        # self.x_ln=nn.LayerNorm(hidden_dim)

    def forward(self, state):
        """
        state: (batch_size, n_teeth, state_dim)
        """
        shape = self.shape_embedding(state[:, :, 36:])
        shape = self.shape_ln(shape)
        shape=self.relu(shape)
        
        state_features = state[:, :, :36]
        
        x = torch.cat([state_features, shape], dim=-1)  # (batch, n_teeth, input_dim)
        # x=state_features
        # print(x.shape)
        x = self.embedding(x)
        
        pos_embed= self.pos_embedding(torch.tensor(list(range(self.n_teeth)),device=state.device))        
        x = x + pos_embed.unsqueeze(0)
        
        
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        
        action = self.fc_out(x) 
        mask=self.weight_out(x)
        mask=self.sigmod(mask/args.temperature)
        mask_action=action*mask
        
        return mask_action,mask


def create_env(sampleFileName=None,teeth_ids=None,shape_dir=None,hull_dir=None,args=None,shape_id=None):
    step_paths=[f for f in os.listdir(sampleFileName) if f.startswith("step")]
    seq_state=[]
    teeth_is_null=[]
    for step in range(1,len(step_paths)+1):
        json_path=os.path.join(sampleFileName,f'step{step}.json')
        teeth28=[]
        with open(json_path,'r') as file:
            data=json.load(file)
            for id in teeth_ids:
                if f'{id}' in data.keys():
                    x,y,z,qx,qy,qz,qw=data[f'{id}']
                    teeth28.append([x,y,z,qw,qx,qy,qz])
                else:
                    teeth28.append([0]*7)
                    if step==1:
                        teeth_is_null.append(len(teeth28)-1)
        seq_state.append(teeth28)
    
    seq_state=np.array(seq_state)
    xyz=seq_state[:,:,:3]
    rotation=seq_state[:,:,3:]
    rotation=utils_np.quat_to_matrix9D(rotation.reshape(-1,4))
    rotation=utils_np.matrix9D_to_6D(rotation).reshape(-1,len(teeth_ids),6)
    seq_state=np.concatenate((xyz,rotation),axis=2)
    seq_state[:,teeth_is_null,:]=teeth_mean[teeth_is_null,:]
    convex_hull=[]
    for id in teeth_ids:
        hull_path=os.path.join(hull_dir,f"{sampleFileName.split('/')[-1]}_{id}.ply")
        if os.path.exists(hull_path):
            hull = o3d.io.read_triangle_mesh(hull_path)
            convex_hull.append(hull)
        else:
            convex_hull.append(None)
    teeth_shape=[]
    for id in teeth_ids:
        shape_path=os.path.join(shape_dir,f"{sampleFileName.split('/')[-1]}-{id}.npy")
        if os.path.exists(shape_path):
            shape=np.load(shape_path)
            teeth_shape.append(shape)
        else:
            teeth_shape.append(np.array([[0]*108]))
    teeth_shape=np.concatenate(teeth_shape)
    teeth_shape_tensor=torch.from_numpy(teeth_shape).to(device=device).to(dtype=torch.float32)
    # if is_train_data:
    # teeth_shape_list.append(teeth_shape_tensor)
    if shape_id is None:
        teeth_shape_list.append(teeth_shape_tensor)
        sampletoid[sampleFileName]=len(teeth_shape_list)-1
        shape_id=len(teeth_shape_list)-1
    first_step=np.concatenate((seq_state[0],seq_state[seq_state.shape[0]-1]),axis=1)
    env=OrthoEnv(first_step.astype(np.float32),seq_state.astype(np.float32),convex_hull=convex_hull,teeth_ids=teeth_ids,
                collision_punishment=args.collision_punishment,alpha_trans=args.alpha_trans,alpha_angle=args.alpha_rotation,
                shape_id=shape_id,alpha_smooth=args.alpha_smooth)
    return env


def f(filename_list):
    valid=[]
    for filename in filename_list:
        if f"{filename.split('/')[-1]}.json" in error_cases:
            # print(f"❌ error_case  {filename}")
            continue
        if os.path.isdir(os.path.join("result_vis/gt_train",filename.split('/')[-1])):
            continue
        valid.append(filename)
    return valid

def make_env_list(filename_list,args=None,shape_id=None):
    env_list=[]
    if len(filename_list)==1:
        env=create_env(filename_list[0],teeth_ids,shape_dir=shape_dir,hull_dir=hull_dir,args=args,shape_id=shape_id)
        env.reset()
        env_list.append(env)
        return env_list
    for filename in tqdm(filename_list):
        if f"{filename.split('/')[-1]}.json" in error_cases:
            print(f"❌ error_case  {filename}")
            continue
        if os.path.isdir(os.path.join("result_vis/gt_train",filename.split('/')[-1])):
            continue

        env=create_env(filename,teeth_ids,shape_dir=shape_dir,hull_dir=hull_dir,args=args)
        env.reset()
        env_list.append(env)
        
    return env_list

def load_expert_data(env_list):
    for env in tqdm(env_list[:],desc="main"):
        obs=env.reset()
        done=False
        n_done=False
        data_list=[]
        while(not done and not n_done):
            n_reward,n_obs,n_done,n_discount,action=env.default_action()
            next_obs,reward,done,_,reward_dict,collision_record,_=env.step(action,is_load_expert=True)
            ####action->(-1,1)####
            action=(action/max_diff).clip(-1.0,1.0)
            data=(obs,action,next_obs,reward,done,n_obs,n_reward,n_done,n_discount,env.shape_id)
            obs=next_obs
            rb.add(data,expert_data=True)
            data_list.append(data)

            # print(obs.shape)       
        obs_np=np.array([data[0] for data in data_list])
        state_normalizer.update(obs_np)

        reward_np=np.array([data[3] for data in data_list]+[data[6] for data in data_list])
        reward_scaler.update(reward_np)

        env.reset()

@torch.no_grad()
def create_agent_data(env,total_reward):
    obs=np.concatenate((env.state.copy(),env.last_action),axis=1)
    init_obs=obs.copy()
    n_reward=0
    data=[]
    for step in range(args.n_step):
        # print("normaized_state shape",normalized_state.shape)
        actions,mask=actor(state_normalizer.normalize(torch.from_numpy(obs).to(device),[env.shape_id]))
        actions += torch.normal(0, torch.ones_like(max_diff_tensor)*get_noise(global_step)).to(device)
        actions = actions.cpu().numpy().clip(-1,1)
        next_obs,reward,done,_,reward_dict,_,truncated=env.step(actions*max_diff,global_step=global_step)

        obs=next_obs
        if args.gamma>0:
            n_reward+=(args.gamma**step)*reward
        else:
            n_reward+=reward
        if len(data) == 0:
            data.extend([init_obs,actions,next_obs,reward,done])
        total_reward+=reward
        
    data.extend([next_obs,n_reward,done,args.gamma**(step+1),env.shape_id])

    ###calcuate td-error####
    normal_state=state_normalizer.normalize(torch.from_numpy(data[2]).to(device),[env.shape_id])
    next_state_actions,_=target_actor(normal_state)
    qf1_next_target = qf1_target(normal_state, next_state_actions)
    next_q_value = torch.Tensor(reward_scaler.normalize(data[3])).to(device).flatten() + (1 - data[4]) * args.gamma * (qf1_next_target).view(-1)
    n_normal_state=state_normalizer.normalize(torch.Tensor(data[5]).to(device),[env.shape_id])
    next_n_actions,_=target_actor(n_normal_state)
    qf1_n_target=qf1_target(n_normal_state,next_n_actions)
    next_n_q_value=torch.Tensor(reward_scaler.normalize(data[6])).to(device).flatten()+(1-data[7])*data[8]*(qf1_n_target).view(-1)
    next_q_value=args.q_alpha*next_q_value+(1-args.q_alpha)*next_n_q_value



    qf1_a_values = qf1(state_normalizer.normalize(torch.from_numpy(data[0]).to(device),[env.shape_id]), torch.Tensor(data[1]).to(device)).view(-1)
    td_error = torch.abs(qf1_a_values-next_q_value).cpu().numpy()

    ####action
    data[1]=data[1].squeeze(0)
    rb.add(tuple(data),error=td_error.item())
    # print(td_error.item())
    length=env.T
    if done or truncated:
        env.reset()
    return done or truncated,total_reward,length,reward_dict


def plot(x,y,label,xlable,ylabel,save_path):
    plt.figure(figsize=(10,8))
    plt.plot(x,y,label=label)
    plt.xlabel(xlabel=xlable)
    plt.ylabel(ylabel=ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(save_path)
    plt.close()

test_reward_list=[[],[],[]]
test_len_list=[[],[],[]]
@torch.no_grad()
def test():
    for i in range(len(test_env_list)):
        env=test_env_list[i]
        obs=env.reset()
        done=False
        test_total_reward=0
        discount=1
        truncated=False
        while (not done) and (not truncated):
            actions,_=actor(state_normalizer.normalize(torch.Tensor(obs).to(device),[env.shape_id]))
            actions = actions.cpu().numpy()
            next_obs,reward,done,_,reward_dict,_,truncated=env.step(actions*max_diff)
            obs=next_obs
            test_total_reward+=discount*reward
            discount*=args.gamma
        test_reward_list[i].append(test_total_reward)
        test_len_list[i].append(env.T)
    test_x=[i*args.plot_frequency for i in range(len(test_reward_list[i]))]
    plt.figure(figsize=(10,8))
    for i in range(len(test_reward_list)):
        plt.plot(test_x,test_reward_list[i],label=f"sample_{i}")
    plt.xlabel("step")
    plt.ylabel("eposide reward")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_root,f"{global_step}_reward_episode.png"))
    plt.close()

    plt.figure(figsize=(10,8))
    for i in range(len(test_reward_list)):
        plt.plot(test_x,test_len_list[i],label=f"sample_{i}")
    plt.xlabel("step")
    plt.ylabel("len")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_root,f"{global_step}_len.png"))
    plt.close()


def get_lambda_mask(cur_step):
    return args.lambda_mask
    # return min(100,50+cur_step/10000)

def get_noise(cur_step):
    # return 0.01
    return max(0.01,0.2-cur_step/1e7)

if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}_beta"
    save_root=f"result_L1_part3/{len(teeth_ids)}_collpun_{args.collision_punishment}_\
lr_{args.actor_learning_rate}_gamma_{args.gamma}_noise_{args.exploration_noise}_{args.hidden_dim}_tem_{args.temperature}_L1_{args.lambda_mask}_L2_{args.alpha_L2}_weightsum_{args.weight_sum}_fromscratch"
    os.makedirs(save_root,exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    sampletoid={}
    teeth_shape_list=[]


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
    filename_list=[os.path.join(args.train_dir,f) for f in os.listdir(args.train_dir) if f.startswith("C")]
    filename_list=f(filename_list)
    env_list=make_env_list(filename_list,args)
    filename_list_test=[os.path.join(args.test_dir,f) for f in os.listdir(args.test_dir) if f.startswith("C")]
    test_env_list=make_env_list(filename_list_test,args)

    state_normalizer=RunningMeanStdState(shape=(28,108+9),teeth_pos=teeth_ids_pos,teeth_shape_list=teeth_shape_list,device=device)
    reward_scaler=Scale(shape=(1,))
    rb=Memory(args.buffer_size,device=device,state_normalizer=state_normalizer,reward_scaler=reward_scaler,weight_expert=args.expert_weight)
    
    load_expert_data(env_list)
    del(env_list)

    actor=Actor(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                n_heads=args.n_head, n_layers=args.n_layers,
                hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    
    target_actor=Actor(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                    n_heads=args.n_head,n_layers=args.n_layers,
                    hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    
    qf1=QNetwork(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                n_heads=args.n_head, n_layers=args.n_layers,
                hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    qf1_target=QNetwork(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                n_heads=args.n_head, n_layers=args.n_layers,
                hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    # initialize_weights(qf1)
    if args.save_path is not None:
        actor_state,q_state=torch.load(args.save_path,map_location=device)
        actor.load_state_dict(actor_state)
        qf1.load_state_dict(q_state)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    # q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.q_learning_rate)
    # actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_learning_rate)
    q_optimizer = optim.AdamW(qf1.parameters(), lr=args.q_learning_rate, 
                              weight_decay=args.q_weight_decay,betas=(0.9,0.999))
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.actor_learning_rate, 
                                weight_decay=args.actor_weight_decay,betas=(0.9,0.999))


    done,n_done=False,False
    env_i=random.randint(0,len(filename_list)-1)
    env_cur=make_env_list([filename_list[env_i]],args,shape_id=sampletoid[filename_list[env_i]])[0]
    env_cur.reset()
    total_reward,total_trans_reward,total_rotation_reward,total_collision_punishment,total_smooth_punishment=0,0,0,0,0
    total_reward_list,trans_list,rotation_list,q_loss_list,actor_loss_list,reward_list,collision_punishment_list,smooth_punishment_list=[],[],[],[],[],[],[],[]
    num_punishment_list,total_num_punishment=[],0
    L1_loss_list,L2_loss_list=[],[]
    for global_step in tqdm(range(args.total_timesteps)):
        if args.anneal_lr:
            frac = 1.0 - (global_step - 1.0) / args.total_timesteps
            q_lrnow = frac * args.q_learning_rate
            actor_lrnow = frac * args.actor_learning_rate
            q_optimizer.param_groups[0]["lr"] = q_lrnow
            actor_optimizer.param_groups[0]["lr"]=actor_lrnow
        done,total_reward,length,reward_dict=create_agent_data(env_cur,total_reward)
        if done:
            env_i=random.randint(0,len(filename_list)-1)
            env_cur=make_env_list([filename_list[env_i]],args,shape_id=sampletoid[filename_list[env_i]])[0]
            env_cur.reset()
            total_reward_list.append(total_reward)
            writer.add_scalar("charts/episodic_return",total_reward,global_step)
            writer.add_scalar("charts/episodic_length",length,global_step)
            total_reward=0

        batch,idxs,is_weight=rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions,_=target_actor(batch.next_obs)
            qf1_next_target=qf1_target(batch.next_obs,next_state_actions)
            next_q_value = batch.reward.flatten() + (1 - batch.done.flatten()) * args.gamma * (qf1_next_target).view(-1)

            n_state_actions,_=target_actor(batch.n_obs)
            qf1_n_target=qf1_target(batch.n_obs,n_state_actions)
            n_q_value=batch.n_reward.flatten()+(1-batch.n_done.flatten())*batch.n_discount*(qf1_n_target).view(-1)

            q_value=args.q_alpha*next_q_value+(1-args.q_alpha)*n_q_value

        qf1_a_values=qf1(batch.obs,batch.action).view(-1)
        td_error=torch.abs(qf1_a_values-q_value).detach().cpu().numpy().flatten()
        qf1_loss=(F.mse_loss(qf1_a_values,q_value,reduction='none')*torch.from_numpy(is_weight).to(device)).mean()
        q_optimizer.zero_grad()
        qf1_loss.backward()
        q_optimizer.step()

        for i in range(len(idxs)):
            rb.update(idxs[i],td_error[i].item())


        
        if global_step % args.policy_frequency == 0:
            action,mask=actor(batch.obs)
         
            # L1_loss=get_lambda_mask(global_step)*mask.abs().mean()
            upper = F.relu((mask[:, :14, :].reshape(-1, 14).sum(dim=-1) - args.weight_sum)).mean()
            lower = F.relu((mask[:, 14:, :].reshape(-1, 14).sum(dim=-1) - args.weight_sum)).mean()
            L1_loss = get_lambda_mask(global_step) * (upper + lower)
            
            L2_loss=args.alpha_L2*((mask*(1-mask)).mean())
            # L2_loss=np.abs(0)
            
            actor_loss = -qf1(batch.obs, action).mean()+L1_loss+L2_loss
            # actor_loss = -qf1(batch.obs, action).mean()+L1_loss
            # actor_loss = -qf1(batch.obs, action).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        total_rotation_reward+=reward_dict["rotation_reward"]
        total_trans_reward+=reward_dict["trans_reward"]
        total_collision_punishment+=reward_dict["collision_punishment"]
        total_smooth_punishment+=reward_dict["smooth_punishment"]
        total_num_punishment+=reward_dict["num_punishment"]
        if (global_step)%args.plot_frequency==0:
            print(reward_dict)
            trans_list.append(total_trans_reward/(global_step+1))
            rotation_list.append(total_rotation_reward/(global_step+1))
            collision_punishment_list.append(total_collision_punishment/(global_step+1))
            smooth_punishment_list.append(total_smooth_punishment/(global_step+1))
            num_punishment_list.append(total_num_punishment/(global_step+1))
            q_loss_list.append(qf1_loss.item())
            actor_loss_list.append(actor_loss.item())
            reward_list.append(batch.reward.mean().item())
            L1_loss_list.append(L1_loss.item())
            L2_loss_list.append(L2_loss.item())

            x=[i*args.plot_frequency for i in range(len(trans_list))]
            plot(x,trans_list,label="tran",xlable="step",ylabel="tran_reward",save_path=os.path.join(save_root,f"{global_step:04d}_trans.png"))
            plot(x,rotation_list,label="angle",xlable="step",ylabel="rotation_reward",save_path=os.path.join(save_root,f"{global_step:04d}_rotation.png"))
            plot(x,collision_punishment_list,label="collision_punishment",xlable="step",ylabel="collision_punishmet",save_path=os.path.join(save_root,f"{global_step:04d}_collision.png"))
            plot(x,q_loss_list,label="q_loss",xlable="step",ylabel="q_loss",save_path=os.path.join(save_root,f"{global_step:04d}_q_loss.png"))
            plot(x,actor_loss_list,label="actor_loss",xlable="step",ylabel="actor_loss",save_path=os.path.join(save_root,f"{global_step:04d}_actor_loss.png"))
            plot(x,reward_list,label="reward_mean",xlable="step",ylabel="step_reward",save_path=os.path.join(save_root,f"{global_step:04d}_reward_mean.png"))
            plot(x,smooth_punishment_list,label="smooth",xlable="step",ylabel="smooth",save_path=os.path.join(save_root,f"{global_step:04d}_smooth.png"))
            plot(x,num_punishment_list,label="num_moved",xlable="step",ylabel="num_moved",save_path=os.path.join(save_root,f"{global_step:04d}_num_moved.png"))
            plot(x,L1_loss_list,label="L1_loss",xlable="step",ylabel="L1_loss",save_path=os.path.join(save_root,f"{global_step:04d}_L1_loss.png"))
            plot(x,L2_loss_list,label="L2_loss",xlable="step",ylabel="L2_loss",save_path=os.path.join(save_root,f"{global_step:04d}_L2_loss.png"))
            test()

        if args.save_model and global_step%99999==0:
            model_path = f"{save_root}/{global_step}.cleanrl_model"
            torch.save((actor.state_dict(), qf1.state_dict()), model_path)
       



   

 

    
    
