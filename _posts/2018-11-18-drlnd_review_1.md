---
layout:     post
title:      "Review: Introduction to Deep Reinforcement Learning"
date:       2018-11-18 10:00:00
author:     "Jun"
img: 20181118.jpg
tags: [python, Deep Reinforcement Learning]
---
*source:TripAdvisor*

# Introduction to Deep Reinforcement Learning

강화학습이 유행이다. 강화학습 봇이 아타리도 깨고 슈퍼마리오도 깨고 퀘이크도 깬다. 어떻게 하면 나도 봇을 만들 수 있을까.

<img src='/assets/materials/20181118/deepmind.gif' />

수학의 정석을 펼치면 행렬이 나오듯 강화학습 책을 열면 먼저 MDP(Markov Decision Process)가 기다리고 있다.

<img src='/assets/materials/20181118/AAMarkov.jpg' />
*러시아의 수학자 안드레 안드레비치 마르코프*

강화학습 공부를 여러번 시도했지만 MDP가 뭐지? -> 마르코프 프로세스가 뭐지? -> 전이확률이 뭐지? -> 다음에 알아보자..
의 과정을 거친 어두운 과거가 있었다.

이번에는 이론보다 먼저 문제부터 접근해보자.

## OpenAI GYM

그동안 공부해봤던 딥러닝은 인풋-타겟 데이터를 준비하고 이를 사용해 네트워크를 학습시키기만 하면 되었다.  

강화학습은 조금 다르다. 학습 시키는 것은 같지만, 인풋과 타겟을 준비해두는 개념은 아니다.  

예를 들어 미로를 탈출하는 봇을 학습시킨다고 하면, 미로 환경을 먼저 준비해야 한다.   

그 미로 환경안에서 우리의 봇은 실패와 성공을 거듭하며 탈출하는 방법을 익히게 된다.  

즉, 강화학습을 하려면 환경과 에이전트(봇)을 준비해야 하는데 이 분야에서 가장 많이 활용되는 라이브러리는 OpenAI에서 제공하는 `gym`이다. 


```python
import gym
```

`gym`은 여러 편리한 환경들을 미리 만들어두었다. 우리는 그것을 가져다 쓰기만 하면 된다.  

처음 배우자마자 스타크래프트 AI나 퀘이크 AI를 만들 수는 없다.  

가장 만만해보이는 녀석을 하나 골라보자.  


```python
env = gym.make('CliffWalking-v0')
```

<a href="https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py">`CliffWalking`</a> 문제는 아래와 같다.


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

cliff_env = np.ones(env.shape)
cliff_env[3, :] = -1
cliff_env[3, 0] = 0
cliff_env[3, -1] = 0
fig, ax = plt.subplots(figsize=env.shape[::-1])
sns.heatmap(cliff_env, linewidths=1, ax=ax, cmap='RdBu')

ax.text(0.2, 3.5, "START")
ax.text(11.3, 3.5, "END")

for i in range(10):
    ax.text(i+1 + 0.2, 3.5, "CLIFF", color='white')
plt.show()
```


    <Figure size 1200x400 with 2 Axes>


주인공 에이전트의 미션은 좌하단 스타팅에서 우하단 도착지까지 이동하는 것이다.  

에이전트는 위, 아래, 왼쪽, 오른쪽으로 1칸씩 이동할 수 있으며 이동할때마다 -1점을 받는다.

만약 CLIFF로 표시된 셀로 이동하게 되면 -100점을 받으며 에이전트는 낙사하여 게임을 다시 시작하지.

이 게임의 목적은 에이전트가 살아있는 채로 스타팅에서 도착지로 가장 최적으로 도달하는 방법을 학습하는 것이다.

`env` 변수에 할당한 환경에서 몇가지 정보를 뽑아보자.


```python
# 환경의 크기
env.shape
```




    (4, 12)




```python
# 에이전트가 취할 수 있는 가짓수
env.action_space.n
```




    4




```python
# 스타팅 셀 (36번째 셀)
env.reset()
```




    36



## 우리한테 쉽다고 쉬운 문제는 아니다.

이 문제를 우리가 푸는데 0.1초도 필요하지 않다. 그냥 눈으로 봐도 답은 정해져있다.  

위로 1칸만 올라간다음, 11칸을 직진하고 다시 1칸을 내려오면 된다. 

<img src='/assets/materials/20181118/cliff_human_approach.png' width=400px>

개/고양이 분류하는 딥러닝처럼 사람에게 쉬운 문제가 컴퓨터에게는 매우 어려울 수 있다.

우리의 agent도 마찬가지다.

## 그럼 어떻게 풀어야 할까?

이 문제에서 agent은 매 순간 의사결정을 내려야한다. 상하좌우 4가지 action 중에 하나를 골라야 한다.  

그리고 그 선택에 따라 다음 셀로 이동한 봇은 또다시 선택에 직면한다. 새로운 state에서.

스타팅에 agent 대신 우리가 서서 주변을 둘러본다고 생각해보자. 

위에는 평지가 있고 왼편에는 깎아지듯 떨어지는 절벽이 있다. 그리고 절벽의 먼 너머에는 도착지가 있다.

절벽으로 발을 내딫으면 도착지에 더 가까워지지만 목숨이 날아간다. 평지로 이동하면 거리는 줄어들지 않지만 그래도 목숨은 부지한다.

우리 상식으로는 후자가 전자보다 더 value있는 행위이므로 (죽으면 말짱 소용없으니) 평지로 이동하는 것을 택한다.

이렇듯 우리는 다음 action을 선택할 때 가장 value가 높은 쪽을 선택한다. 

<img src='/assets/materials/20181118/cliff_first_move.jpg' width=400px>


## agent를 움직여보자.

agent는 어떻게 env (게임 환경)과 상호작용할까?  

Cliff Walking 문제에서 가장 처음 시작하는 지점은 36번째 셀이다.


```python
# 스타팅 state
env.reset()
```




    36




```python
env.action_space.n
```




    4



agent가 취할 수 있는 action은 0, 1, 2, 3이며 각 숫자는 다음의 방향으로의 1칸 이동을 의미한다.

- UP = 0
- RIGHT = 1
- DOWN = 2
- LEFT = 3


```python
int2action = {
    0:'UP',
    1:'RIGHT',
    2:'DOWN',
    3:'LEFT'
}
```

위로 이동해보자.


```python
env.step(1)
```




    (36, -100, False, {'prob': 1.0})



env의 `step`함수에 action을 인자로 입력하면, env내의 agent의 위치정보가 갱신되고, 이에 해당하는 4가지 정보가 리턴된다.

- observation(object): action으로 인해 발생하는 관측값으로, 여기서는 다음 시점의 state가 된다.
- reward(float): action으로 인해 얻게 되는 reward값. 강화학습의 목적은 총 reward (value)를 최대화시키는 것이 된다.
- done(boolean): env를 리셋해야하는지에 대한 불리언 값. 강화학습이 episodic task인 경우 done은 episode의 종료를 의미한다. (후에 설명)
- info(dict): 디버깅을 위한 정보로 마지막 state 변화에 대한 raw 확률값을 담는다. 학습에는 사용하지 않는다.

agent는 env안에서 state를 파악하고, action을 실행하여, action에 대한 reward와 다음 state 정보를 얻는다.

강화학습에서는 state 파악과 action 실행을 같은 시점으로, reward와 다음 state 정보 획득을 같은 시점으로 묶는다.

예를 들어 현재 시점을 t라고 하면,

- agent는 env로부터 현재 state 정보 $S_t$를 얻는다.
- agent는 일련의 판단을 통해 action $A_t$를 선택하고 실행한다.
- env는 agent의 action에 따라 reward $R_{t+1}$을 전달한다.
- agent의 action에 따라 새로운 state 정보 $S_{t+1}$이 agent에게 전달된다.


<img src='/assets/materials/20181118/env_agent_interaction.jpg' width=400px>

env가 리셋된 후 첫 스타팅에서는 리워드 정보가 없다.

### value는 어떻게 계산해야 할까?

이처럼 agent는 현재 state에서 action을 수행한다음 env로부터 reward와 새로운 state 정보를 얻는다.

문제를 풀기 위해 agent는 reward의 총합, 즉 value가 가장 큰 쪽으로 움직이도록 학습한다.

그렇다면 value라는 것은 어떻게 계산해야 할까??




<img src='/assets/materials/20181118/first_move.jpg' width=200px>

스타팅에서 갈 수 있는 다음 state는 바로 위와 오른쪽이었다.   

agent는 위 state의 value가 오른쪽 절벽의 value보다 더 높다고 판단한다.   

절벽으로 이동할 때 agent는 -100을, 평지로 이동할때는 -1의 reward를 environment로부터 받는다.  

그렇다면 스타팅 state에서 이동하는 경우 위 state는 -1, 오른쪽 state는 -100의 value를 가지고 있다고 생각할 수 있다.

-1 > -100이므로 바로 다음 reward에 기반한 value 책정은 나쁘지 않은 방법이다.

하지만 코앞의 reward만으로 우리는 의사결정을 내리지 않는다. 

연속적으로 action을 결정해야 하는 상황이라면, 또 지금의 action이 나중의 action에도 영향을 준다면,  

지금 시점의 value는 앞으로 받게 될 reward들의 총합($G_t$)이라고 생각하는 것이 옳다.

$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...$

그런데 시점이 다른 두 reward의 크기가 같다해도, 언제 얻을 수 있느냐에 따라 그 가치가 달라질 수 있다.  

당장 받는 1달러는 내일 받는 1달러보다 가치가 조금 더 높다. 따라서 미래 가치에 대한 discount rate인 $\gamma$를 적용한다. 

discount rate를 적용된 value의 식은 아래와 같다.

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$

$\gamma$는 다음과 같은 특성을 지닌다.

- $0<= \gamma <= 1$ discount rate는 미래 가치를 조금 깎으므로 0과 1사이에 위치한 어떤 값이 된다.
- $\gamma$가 0이 되면 $G_t = R_{t+1}$가 되므로, agent는 바로 다음의 reward만을 신경쓰게 된다.
- $\gamma$가 1이 되면 $G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...$가 되므로, agent는 먼 미래의 reward도 코앞의 reward만큼 중요하게 신경쓴다.
- 즉, $\gamma$가 커지면 커질수록 미래의 reward를 더 신경쓴다고 볼 수 있다.


### One-Step Dynamics

정리하자면, agent는 현재 state에서 실행할 action을 결정해야 한다. 

어떤 action을 선택할까? 돌아오는 return $G_t$가 가장 높은 action을 선택한다.  

앞에서 $G_t$는 당장의 reward 뿐 아니라 먼 미래의 값까지 할인해서 더한 값이라고 정의했다.

즉, agent는 이런 복잡한 연속적인 의사결정의 문제를

지금 이 state에서 할 수 있는 action 중에 value가 가장 높은 action은 뭐지?라는 작은 문제의 연속으로 만들어서 푼다.

이를 one step dynamics라고 한다.

### Finite MDPs

지금까지 설명한 내용이 강화학습의 기본이다.

우리의 agent는 절벽을 무사히 그리고 빠르게 건너는 문제를 푼다. 이 agent가 어떤 action을 선택하느냐에 따라 reward가 달라지고 (-1 or -100), 다음 state가 달라진다. agent는 변화하는 환경과 상호작용하면서 문제를 푸는 방식을 익힌다. 이것이 reinforcement learning, 강화학습이다.

agent는 action을 선택함에 있어 value 최대화를 기준으로 삼는다. value는 단기적인 reward 뿐 아니라 먼 미래의 reward도 discount하여 고려한다. 즉, agent는 expected cumulative reward를 최대화하며 문제를 푸는 것을 목표로 학습한다.

agent가 활동하는 environment는 one-step dynamics라는 특징을 가진다. t-1시점의 정보는 t시점으로, t시점의 정보는 t+1시점으로 사슬고리가 연결되듯 environment가 구성되어있다.

이것이 바로 MDP의 속성이다. Cliff Walking 문제의 state와 action은 특히 그 경우의 수가 한정되어 있으므로 Finite MDP라고 할 수 있다. 

Finite MDPs는 다음과 같은 특징을 가진다.

- $S$ = a finite set of states
- $S^+$ = a finite set of states (episodic task)
- $A$ = a finite set of actions
- $R$ = a set of rewards
- one-step dynamics
- $\gamma$ = the discount rate

문제가 시작과 끝이 있으면 episodic task, 끝이 없이 계속 되는 문제를 continuous task라고 한다. 

## Policy

강화학습을 푼다는 것은, 연속적으로 의사결정을 내려야 하는 상황에서 최적의 의사결정을 내리는 추상적인 전략을 깨우친다는 것과 같다.

서점에 가보면 엄청나게 많은 주식투자 책을 찾아볼 수 있다. 단기적인 시장 상황을 통해 매매하는 기술적 투자나 회사의 내재적인 가치를 판단해 저평가 주식을 찾아내는 가치 투자 등등 저마다 다양한 투자 정책을 제시한다.

Cliff Walking로 마찬가지다. 시작점에서 출발한 agent가 종료지점까지 도착하는데는 다양한 길이 있다. 절벽을 따라 걸을 수도 있고, 절벽에서 멀리 떨어져서 걸을 수도 있다.

policy(정책)은 $\pi$라는 함수로 생각하면 된다.   
이 함수에 state $s$를 입력하면 그 policy를 따르는 action $a$를 얻을 수 있다. 

이를 수식으로 표현하면, 
$\pi : S \rightarrow A$라 할 수 있다.

$s$를 넣으면 무조건 policy에 따른 $a$가 100% 출력되는 policy를 deterministic policy라고 한다.

그런데 우리 인생이 항상 의도한대로 흘러가지 않듯, 오른쪽으로 가려는 agent도 강한 바람에 위로 이동할 수도 있다.

이렇게 확률적 요소가 반영된 policy를 stochastic policy라고 한다. 

이 함수는 state $s$에서 action $a$를 수행할 확률(0에서 1사이)을 리턴한다.

즉, 수식으로 표현하면
$\pi : S \times A \rightarrow [0, 1]$

### Policy 구현

Cliff Walking에서 각 state는 4가지 action을 수행할 수 있다. 어떤 state s에서의 policy를 각 스테이트가 가질 수 있는 4가지 액션값이 발생할 수 있는 확률 값을 담은 array라고 생각해보자.

두가지 policy를 구현한 후 각 policy의 value를 살펴보자.

- random_policy: 어떤 state에 있든 4가지 액션 중 하나를 랜덤하게 선택한다. (각 25%의 확률)
- optimal_policy: 위로 한칸 이동한 후 끝까지 오른쪽으로 가서 한칸 아래로 가는 하드코딩된 최적 policy


```python
import numpy as np
from collections import defaultdict
import sys
```


```python
def random_policy(env, state):
    """state에 관계없이 policy_s는 4가지 액션이 0.25의 확률을 갖는 array를 리턴한다."""
    
    policy_s = np.ones(env.nA)  / env.nA
    
    return policy_s
```


```python
policy = random_policy(env, 12)
np.random.choice(np.arange(env.nA), p=policy)
```




    0




```python
def optimal_policy(env, state):
    """하드코딩된 optimal policy"""
    
    if state == 36:
        return [1.0, 0.0, 0.0, 0.0]
    elif state != 35:
        return [0.0, 1.0, 0.0, 0.0]
    elif state == 35:
        return [0.0, 0.0, 1.0, 0.0]
    else:
        return [0.25, 0.25, 0.25, 0.25]
```

생성한 policy를 실행하는 시뮬레이션 함수를 만들어본다. 

시뮬레이션 함수는 첫 스타팅 state 이후 입력받은 policyFunction에 의해 다음 action를 선택하고 실행한다. 

action을 실행하여 reward와 새로운 next_state를 얻는다.

next_state가 종착지이거나, episode의 최대 길이에 도달하는 경우 게임을 종료한다.
(이 구현에서는 절벽에 떨어져 -100을 받더라고 게임이 종료된 것으로 간주하지 않는다)

그렇지 않은 경우, next_state는 현재 state가 되고 다시 policy에 따른 action을 선택한다.


```python
def run_simulation(env, policyFunction):
    """"""
    
    MAX_EPISODE_LENGTH = 30
    
    # value를 쌓을 score를 정의한다.
    score = 0
    
    # state를 리셋하고 스타팅 state를 얻는다.
    state = env.reset()
    
    for i in range(MAX_EPISODE_LENGTH):
        
        # 주어진 state에서 policyFunction을 따르는 action의 확률값을 얻는다.
        policy = policyFunction(env, state)
        
        # action의 확률값을 사용해 action을 선택한다.
        action = np.random.choice(np.arange(env.nA), p=policy)
    
        # action을 실행하여 다음 state, reword, done 여부를 얻는다.
        next_state, reward, done, info = env.step(action)
        
        # 현재 score에 reward를 더해 업데이트한다.
        score += reward
        
        print("\r v: {} | s: {} | a: {} | r: {} | next s: {} | done: {}".format(score, state, int2action[action], reward, state, done))
        sys.stdout.flush()
        if done or i == MAX_EPISODE_LENGTH - 1:
            print("simulation finished with value = {}.".format(score))
            break
            
        else:
            state = next_state
        
run_simulation(env, random_policy)
```

     v: -1 | s: 36 | a: LEFT | r: -1 | next s: 36 | done: False
     v: -2 | s: 36 | a: UP | r: -1 | next s: 36 | done: False
     v: -3 | s: 24 | a: DOWN | r: -1 | next s: 24 | done: False
     v: -4 | s: 36 | a: DOWN | r: -1 | next s: 36 | done: False
     v: -5 | s: 36 | a: DOWN | r: -1 | next s: 36 | done: False
     v: -105 | s: 36 | a: RIGHT | r: -100 | next s: 36 | done: False
     v: -205 | s: 36 | a: RIGHT | r: -100 | next s: 36 | done: False
     v: -206 | s: 36 | a: DOWN | r: -1 | next s: 36 | done: False
     v: -207 | s: 36 | a: DOWN | r: -1 | next s: 36 | done: False
     v: -307 | s: 36 | a: RIGHT | r: -100 | next s: 36 | done: False
     v: -308 | s: 36 | a: DOWN | r: -1 | next s: 36 | done: False
     v: -309 | s: 36 | a: UP | r: -1 | next s: 36 | done: False
     v: -310 | s: 24 | a: LEFT | r: -1 | next s: 24 | done: False
     v: -311 | s: 24 | a: LEFT | r: -1 | next s: 24 | done: False
     v: -312 | s: 24 | a: RIGHT | r: -1 | next s: 24 | done: False
     v: -412 | s: 25 | a: DOWN | r: -100 | next s: 25 | done: False
     v: -512 | s: 36 | a: RIGHT | r: -100 | next s: 36 | done: False
     v: -513 | s: 36 | a: UP | r: -1 | next s: 36 | done: False
     v: -514 | s: 24 | a: UP | r: -1 | next s: 24 | done: False
     v: -515 | s: 12 | a: LEFT | r: -1 | next s: 12 | done: False
     v: -516 | s: 12 | a: DOWN | r: -1 | next s: 12 | done: False
     v: -517 | s: 24 | a: RIGHT | r: -1 | next s: 24 | done: False
     v: -518 | s: 25 | a: UP | r: -1 | next s: 25 | done: False
     v: -519 | s: 13 | a: UP | r: -1 | next s: 13 | done: False
     v: -520 | s: 1 | a: LEFT | r: -1 | next s: 1 | done: False
     v: -521 | s: 0 | a: DOWN | r: -1 | next s: 0 | done: False
     v: -522 | s: 12 | a: UP | r: -1 | next s: 12 | done: False
     v: -523 | s: 0 | a: UP | r: -1 | next s: 0 | done: False
     v: -524 | s: 0 | a: LEFT | r: -1 | next s: 0 | done: False
     v: -525 | s: 0 | a: DOWN | r: -1 | next s: 0 | done: False
    simulation finished with value = -525.



```python
run_simulation(env, optimal_policy)
```

     v: -1 | s: 36 | a: UP | r: -1 | next s: 36 | done: False
     v: -2 | s: 24 | a: RIGHT | r: -1 | next s: 24 | done: False
     v: -3 | s: 25 | a: RIGHT | r: -1 | next s: 25 | done: False
     v: -4 | s: 26 | a: RIGHT | r: -1 | next s: 26 | done: False
     v: -5 | s: 27 | a: RIGHT | r: -1 | next s: 27 | done: False
     v: -6 | s: 28 | a: RIGHT | r: -1 | next s: 28 | done: False
     v: -7 | s: 29 | a: RIGHT | r: -1 | next s: 29 | done: False
     v: -8 | s: 30 | a: RIGHT | r: -1 | next s: 30 | done: False
     v: -9 | s: 31 | a: RIGHT | r: -1 | next s: 31 | done: False
     v: -10 | s: 32 | a: RIGHT | r: -1 | next s: 32 | done: False
     v: -11 | s: 33 | a: RIGHT | r: -1 | next s: 33 | done: False
     v: -12 | s: 34 | a: RIGHT | r: -1 | next s: 34 | done: False
     v: -13 | s: 35 | a: DOWN | r: -1 | next s: 35 | done: True
    simulation finished with value = -13.


랜덤 policy는 종착지에 도착하지 못했을 뿐더러 value도 크게 낮은 값을 기록한다.  

반대로 하드코딩한 optimal policy는 마지막 done이 True이고, 누적 값 역시 -13으로 agent가 얻을 수 있는 최대값을 얻었다.

우리의 목표는 agent가 opimal policy를 얻도록 학습하는 것이다.

## 무엇이 Optimal Policy일까?

위에서 구한 랜덤 policy와 optimal policy는 무슨 차이가 있었을까? optimal policy는 누적 값이 랜덤 policy보다 높았다.

누적 값은 각 state에서 얻을 수 있는 reward의 총합이었다. 그럼 누적값이 가장 큰 policy가 optimal policy일까?

앞서 강화학습 문제를 Finite Markov Decision Process(Finite MDP)로 정의했다.

MDP의 특징 중 하나는 One-Step Dynamics다. 해당 state에서의 의사결정은 다음 state와의 관계속에서 이루어진다.

멀리 있는 누적값을 있는 그대로 고려하는 방식이 아니다.

즉, 각 state에서 policy를 통해 얻을 수 있는 value를 고려해야 한다.  

모든 state에서 다른 policy보다 얻을 수 있는 value가 큰 policy를 optimal policy라 한다.

## State-Value Functions

두 policy ($\pi_1$, $\pi_2$)가 있다고 하자. 둘 중 어느 policy가 낫다고 할 수 있을까? 

임의의 state $s$에서 두 policy의 value를 비교해보자. 그리고 모든 state에서 한 policy가 다른 policy보다 value가 크다면, 그 policy는 더 좋다고 할 수 있다.

어떤 state $s$에서 어떤 policy $\pi$가 가지는 value을 구하는 함수를 state-value function이라고 하며 이를 $v_{\pi}(s)$로 표현한다.

이를 수식으로 표현하면,
$v_{\pi}(S) \doteq E_{\pi}[G_t | S_t = s]$

즉, policy $\pi$를 따를때 state $s$의 값은 해당 state가 가질 수 있는 value G_t의 기댓값이라 할 수 있다.

기대값인 이유는, G_t가 확률변수이기 때문이다.


## Bellman Expectation Equation

앞서 value를 계산하는 파트에서 $G_t$를 다음과 같이 정의하였다.

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma R_{t+3} + ..$

그런데 $G_t$는 다른 방식으로도 표현할 수 있다.

Cliff Walking 문제를 다음과 같이 간단히 바꾸어본다.

- state는 3개로 s0 -> s1 -> s2로 agent가 이동한다.
- state를 이동할때 reward는 -1을 받는다.
- discount rate $\gamma$는 1로 설정한다.

이는 아래 그림과 같다.

<img src='/assets/materials/20181118/simple_pic1.jpg' width=400px>

여기서 우리는 마지막 state에 도달했을때 얻는 value를 확실히 안다. 마지막 state에 도달하면 아무 reward를 받지 않고 이후에 받을 reward도 없으므로 이때의 value는 0이다.

이 간단한 문제에서 s2는 s1에서부터 이동하므로, s1의 value는 역산하여 구할 수 있다.
$v_{\pi}(s1) = -1 + 0 = -1$

<img src='/assets/materials/20181118/simple_pic2.jpg' width=400px>

또 s1의 value를 앎으로써 s0의 value도 알 수 있다. 
$v_{\pi}(s1) = -1 + -1 = -2$

<img src='/assets/materials/20181118/simple_pic3.jpg' width=400px>

즉 $v_{\pi}(s)$는 바로 다음에 얻게 될 immediate reward와 다음 state의 value와 같다.

$v_{\pi}(s) = R_{t+1} + v_{\pi}(s_{t+1})$

이를 좀 더 일반화해서 적용해보자.
일반적으로 state는 여러 액션을 가질 수 있으므로 다음 reward와 그 다음 state의 value는 확률변수가 된다. 또 다음 state의 value에는 discount가 적용되므로
위 식에 $\gamma$를 곱하고 확률변수의 기댓값 $E$를 씌우면 아래와 같은 식이 된다.

$$v_{\pi}(s) = E[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s]$$

이것이 Bellman expectation equation이다.

이로서 state가 가지는 value인 $G_t$를 reward와 다음 state의 value로 구할 수 있게 되었다. 

policy $\pi$가 deterministic하다면 공식은 다음과 같이 바꿔쓸 수 있다.

$$v_{\pi}(s) = \Sigma_{s^\prime \in S^+, r \in R} p(s^\prime, r|s, \pi(s))(r+ \gamma v_{\pi}(s^\prime)$$

$\pi(s)$는 state s에 대한 policy에 따른 action이 된다. 
즉, $p(s^\prime, r|s, \pi(s))$는 state s와 action a가 주어졌을 때 새로운 state $s\prime$과 reward r이 리턴될 확률이다. 
그 다음은 그 action에 대한 immediate reward r과 다음 state $s\prime$의 discounted된 state value를 더한 보상의 총합이 된다. 

이를 에피소드의 모든 state $S^+$와 모든 Reward에 대해서 다 합치면 $v_{\pi}(s)$를 구할 수 있다.

만약 action이 확률적으로 결정되는 stochastic policy라면 $\Sigma$에 action이 하나 더 추가된다. 
그리고 뒷단에 $\pi(a|s)$ (state $s$가 주어졌을 때 policy $\pi$가 action $a$를 선택할 확률)가 추가된다.

$$v_{\pi}(s) = \Sigma_{s^\prime \in S^+, r \in R, a \in A} \pi(a|s)p(s^\prime, r | s, a)(r + \gamma v_{\pi}(s^\prime))$$


Bellman Equation은 이것 말고도 3개가 더 있는데, 모두 value function이 재귀적인 관계임을 보여준다.

앞서 살펴보았던 MDP의 One-Step Dynamics (앞뒤로 위치한 state간에 가지는 관계)를 생각해보면 Bellman Equation과 맞닿아있다.

## Action-Value Functions

앞서 각 state에서의 value를 구할 수 있는 state value function에 대해 알아보았다. value는 state에 따라서도 다르지만 각 state에서 취하는 action에 따라서도 달라질 수 있다. 각 state의 action별로 value를 구할 수 있는 function이 action-value function이다.

action value function은 state $s$와 action $a$를 인자로 받아 policy $\pi$를 따를때의 value를 리턴한다. 이를 $q_\pi$로 표기한다.

$$q_{\pi}(s, a) \doteq E_{\pi}[G_t | S_t = s, A_t = a]$$

모든 최적의 policy들은 같은 action-value function $q^*$를 가지며 이를 optimal action-value function이라고 한다.

state-value function $v$와 마찬가지로, $\pi^\prime$을 따르는 모든 state와 action에서의 value가 $\pi$보다 클때 $\pi^\prime$을 최적의 policy $\pi^*$로 정의한다.

action-value function은 어떻게 구현할까?

Cliff Walking 문제에서 각 state에는 4가지 action 옵션이 존재한다. action-value function은 각 state를 key로, 4가지 action의 value값의 array를 value로 하는 dictionary로 구성해볼 수 있다.


```python
Q = defaultdict(lambda x: np.zeros(env.action_space.n))
Q[36] = [10, 100, -1, 5]
```

이 문제에서 Q는 state x action 크기의 테이블로 볼 수 있다. 처음에는 모든 값이 0으로 시작하지만, 차츰 학습을 해나가면서 Q 테이블을 업데이트할 수 있다.


```python
np.argmax(Q[36])
```




    1



## Summing up

지금까지 우리는 강화학습의 문제를 정의하고, 어떻게 풀지를 알아보았다.

강화학습은 sequential한 의사결정을 해야 하는 상황에서 environment와 agent간의 상호작용을 통해 문제를 푼다.

sequential한 의사결정의 문제를 우리는 Markov Decision Process로 정의하여 푼다.

MDP는 One-Step dynamics라는 성질을 가진다. 현재 state와 거기서의 action이 다음 시점의 reward와 next state로 이어진다.

state와 action의 가짓수가 유한한 형태를 Finite MDP라고 한다.

주어진 state에서 어떤 action을 할 것인가를 결정하는 정책을 policy라고 한다.

강화학습은 문제를 잘 푸는 policy를 얻는 것이다. 

좋은 policy란 각 state 관점에서 얻을 수 있는 value가 가장 큰 policy다.

지금 action을 함으로써 얻는 reward만이 value를 설명할 수 없다.

앞으로 받게 될 미래의 불확실한 reward까지도 고려해야 한다.

미래의 값은 아직 실현되지 않았으므로, discount rate $\gamma$를 곱해 더한다.

optimal policy를 얻기 위해 우리는 우리의 policy가 각 state에서 어떤 value를 가지는지 평가해야 한다.

이를 위해 state-value function $v_{\pi}(s)$를 구한다.

각 state에서의 value가 가장 큰 policy가 optimal policy $\pi^*$가 된다.

MDP의 One-Step Dynamics에 따라 각 state에서의 value $G_t$는 바로 다음의 reward와 다음 state에서의 discounted value로 표현할 수 있다. 이 확률 변수에 기댓값을 씌운 것이 Bellman Expectation Equation이다.

$$v_π(s)=E[R_{t+1}+γv_π(S_{t+1})|S_t=s]$$

그런데 state뿐만 아니라 그 state에서의 action에 따라 value가 달라질 수 있다.

policy $\pi$를 따를때 state와 action의 value를 리턴하는 함수를 action-value function $q_{\pi}$라고 한다. 

optimal policy는 모두 같은 q함수를 가진다. 

q함수만 확보하면 그 다음부터는 각 state에서 value를 최대화하는 action을 선택하여 실행하기만 하면 된다. 그것이 바로 optimal policy가 된다.

## Monte Carlo Methods

우리의 agent는 처한 state에서 action을 해나가면서 reward를 얻는다. (interaction)

이를 통해 어떤 state에서 어떤 action을 하면 어느 정도의 value를 얻는지 학습한다. (state-value function or action-value fucntion)

그리고 각 state나 state-action에서 얻을 수 있는 value가 가장 큰 선택을 한다. (choosing action that maximises return value)

그러니까 우리는 정확한 state-value function이나 action-value function을 얻기만 하면 된다. 그 다음에는 argmax하면 되니까.

어떤 state나 state-action 페어가 어떤 value를 가질지 예측하는 것을 prediction problem이라고 한다. 그리고 그를 통해 얻은 $v$나 $q$에서 행동을 하는 것을 control problem이라고 한다.

Finite MDP 형식의 강화학습 문제를 풀 때, policy $\pi$를 따르는 action-value function $q_{\pi}$(Q table)를 추정한다. 

대체 값을 어떻게 추정할 수 있을까?

위에서 우리는 $G_t$를 $R_{t+1} + \gamma q_{\pi}(s, a) (s \in S, a \in A)$와 같은 형식으로 정의했다.

즉, 지금 state에서의 value은 바로 다음에 얻게 될 reward와 그 다음 시점의 value인데, 가보지 않은 미래의 value를 어떻게 알 수 있다는 말이지?

또 문제를 복잡하게 만드는 것은 model이다. 위로 이동한다고 할때 강풍이 불거나 실족하는 등 낮은 확률도 오른쪽으로 이동할 수 있다. 실수할 확률을 안다면 계산할 수 있겠지만 실제 문제를 풀 때 확률을 알 수 있는 경우는 드물다.

state간 전이 확률(model)을 모르는 경우 어떻게 value를 적절하게 측정할 수 있을까? 이때 사용할 수 있는 방법이 Monte Carlo Prediction이다.

Monte Carlo Methods는 간단히 표현하자면 닥치고 돌려보는 것이다. 강풍이 불고 실족을 할 미지의 가능성이 조금 있더라도, 우리가 엄청나게 많이 가보고 그 평균을 내면 정답에 가깝게 다가갈 수 있다.

<a href="https://jsideas.net/montecarlo_visualisation/">Monte Carlo Approximation</a>

Monte Carlo Methods에는 두가지 방법이 있다.

Cliff Walking 문제에서 우리는 갔던 state를 재방문할 수도 있다. 또 그 state에서 행했던 action을 같은 episode내에서 반복할 수도 있다. 

First-visit MC는 무조건 처음 방문한 (s, a)만을 사용해서 q함수의 값을 업데이트한다.

Every-visit MC는 방문한 (s, a)의 value들의 평균을 사용해서 q함수의 값을 업데이트한다.

Every visit MC를 구현해보자.


```python
def random_policy(env):
    return np.ones(env.action_space.n) / env.nA

def generate_episode(policy, env):
    
    MAX_EPISODE_LENGTH = 500
    
    s = env.reset()
    i = 0
    record = []
    
    while True:
        i += 1
        a = np.random.choice(env.nA, p=policy)
        next_s, r, done, info = env.step(a)
        record.append([s, a, r])
        
        if done or i == MAX_EPISODE_LENGTH:
            break
        else:
            s = next_s
    
    return record
```


```python
# policy = random_policy(env)

# generate_episode(policy, env)
```


```python
def mc_prediction(policy, env, gamma):
    
    # Q table
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # s, a 발생횟수
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # return sum
    return_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    
    NUMBER_OF_EPISODES = 500
    
    for i_episode in range(NUMBER_OF_EPISODES):
        
        episode = generate_episode(policy, env)
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for t, state in enumerate(states):
            
            action = actions[t]
            rewards_from_t_plus_1 = rewards[t:]
            corresponding_discounts = discounts[:-(1+t)]
            return_sum[state][action] += sum(rewards_from_t_plus_1 * corresponding_discounts)
            N[state][action] += 1.0
            Q[state][action] = return_sum[state][action] / N[state][action]
        
        print("\repisode {} done with value(s_0): {}".format(i_episode, sum(rewards * discounts[:-1])), end="")
        sys.stdout.flush()
    
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q
    
policy = random_policy(env)         
policy, Q = mc_prediction(policy, env, gamma=0.99)
```

    episode 499 done with value(s_0): -1270.8386261637955


```python
def visualize_policy(policy):

    cliff_env = np.ones(env.shape)
    cliff_env[3, :] = -1
    cliff_env[3, 0] = 0.5
    cliff_env[3, -1] = 0.5
    fig, ax = plt.subplots(figsize=env.shape[::-1])
    sns.heatmap(cliff_env, linewidths=1, ax=ax, cmap='RdBu')

    for k, v in policy.items():
        
        rowNum = k // 12
        colNum = k % 12
        ax.text(colNum + 0.2, rowNum + 0.5, int2action[v], color='white')

    plt.show()
    
visualize_policy(policy)
```


![png](/assets/materials/20181118/core_curriculum_1_47_0.png)


스타팅 state에서 시작해보면 뭔가 절벽으로는 안가는 것 같지만 학습이 잘 되었다고 보기도 어렵다. 그러나 완전 랜덤인 policy만을 사용해서 500번 시뮬레이션한 다음 뽑은 결과로는 나쁘지 않아보인다. 어쨌든 절벽은 피하니까.

## Greedy Policies vs. Epsilon-Greedy Policies

위에서는 상하좌우 중 하나를 랜덤하게 고르는 policy를 MC 시뮬레이션에 넣어 돌려보고 그에 따른 Q함수(테이블)을 구할 수 있었다.

그 Q함수 테이블은 state의 갯수 x action의 갯수만큼의 크기를 가지는데, 각 state별로 가장 value가 큰 action을 골라 위에서 어떤 행동을 해야 할지 결정했다.

각 state에서 value가 가장 큰 action이 바로 greedy action이다. 또 value를 최대화시키는 action을 고르는 것을 greedy policy를 따른다고 한다.

greedy policy는 그럴듯 해보이지만, greedy라는 수식어가 붙는 다른 표현들처럼 그렇게 긍정적이지만은 않다.

새로 이사간 마을에서 근처 밥집을 찾는다고 생각해보자. 이사를 왔으므로 정보가 없어 레스토랑에 대한 value는 모두 0인 상태다. 여기서 운좋게 그럭저럭 적당한 밥집을 찾아 적당히 식사를 즐기게 된다면, 이 행동에는 +점수가 부여된다. 

이를 바탕으로 value 함수를 업데이트하게 되면 주어진 상태에서 최고의 value를 찾아가는 greedy policy로는 그 식당만 계속 가게 된다. 그렇게 되면 그 옆에 있는 최고의 맛집은 놓치게 되는 셈.

즉, 어느정도 최적화된 액션을 찾았다하더라도 어느정도 다른 선택을 할 필요가 있다. 이것이 바로 Epsilon-Greedy Policy $\epsilon-greedy$ policy다.

e-greedy policy는 아주 작은 확률 $\epsilon$로 랜덤한 액션을 택한다. 반대의 경우인 $1-\epsilon$의 확률로 greedy action을 선택한다.

즉 주어진 state에서 가장 value가 높은 값에 대한 action은 $1-\epsilon + {\epsilon \over nA}$의 확률로 선택되고
나머지 액션들은 $nA \over \epsilon$만큼의 선택확률을 갖게 된다.

간단히 파이썬 함수로 구현해보면 다음과 같다.


```python
def get_prob(Q_s, env, epsilon):
    non_greeey_prob = epsilon / env.nA
    policy_s = np.ones(env.action_space.n) * non_greeey_prob
    
    greedy_idx = np.argmax(Q_s)
    policy_s[greedy_idx] += 1 - epsilon
    
    return policy_s
    
Q_s = [10, 1, 2, -3]
get_prob(Q_s, env, 0.05)
```




    array([0.9625, 0.0125, 0.0125, 0.0125])



## MC Control
prediction problem은 state-action이 가지는 value를 추정하는 문제였다.

control problem은 env와의 interaction을 통해 optimal policy $\pi^*$를 결정하는 문제다.

Monte Carlo control method는 Q함수를 추정하는 policy evaluation과 optimal policy를 찾아나가는 policy improvement를 번갈아가면서 계속한다.

## Exploration vs. Exploitation

모든 강화학습 agent는 Exploration-Exploitation Dilemma를 겪는다. 좋은 전략을 찾게 되면 그것을 활용해야 하지만 (exploit) 어느정도 더 나은 전략을 찾기 위한 탐색(explore)을 계속해야 한다. 그 밸런스를 찾아야 함.

MC Control을 통해 optimal policy에 도달하기 위해서는 Greedy in the Limit with Infinite Exploration(GLIE)라는 조건을 만족해야 함.

- 모든 state-action pair s, a는 무한히 많은 횟수로 반복되어야 하며
- policy는 Q함수를 사용해 greedy한 action을 수행하는 policy로 수렴해야 한다.

이러한 조건이 만족되면 agent는 모든 타임 스텝에서 explore를 포기하지 않게 되고, 시간이 갈수록 기존에 쌓은 지식을 더 exploit하게 된다. 

이 조건을 만족하기 위해서 보통 $\epsilon-greedy$ policy의 $\epsilon$값을 점차 수정하면 된다. 예를 들어 $\e$값은 무조건 0보다 큰 값을 유지하되, step i에 따라서 조금씩 discount를 해나가면, i가 무한해짐에 따라 $\epsilon_i$는 0에 근접해진다. 예를 들어 $e_i = {1 \over i}$로 설정할 수 있겠다.

실 구현할때는 얘기가 좀 다름. 수학적으로 convergence가 검증되지 않았더라도 보통 이런 방식을 통해 더 나은 결과를 얻을 수 있다.

- fixed $\epsilon$
- 작은 양의 실수에 이를때까지만 discount하기 (ex 0.1)

이렇게 하는 이유는, 너무 빨리 $\epsilon$을 깎아버리면 나중 episode에서는 새로운 시도를 그만큼 덜하게 된다.
DQN 논문에서는 첫 1백만 프레임에서는 1.0부터 0.1까지 리니어하게 깎고, 그 이후로는 0.1을 픽스해서 적용했다고 한다.

## Incremental Mean

MC는 수많은 에피소드 샘플을 생성한다음, 그 값을 평균내어 state action 페어의 value를 추정한다.

예를 들어 학습을 위해 총 4개의 epsiode를 생성했다고 가정하자. 그리고 그 episode를 하나씩 돌면서 어떤 임의의 state action 페어의 return값을 사용해 Q value를 업데이트한다고 생각해보자.

episode의 순번을 N, 각 순번의 에피소드에서 s, a 페어에 해당하는 return이 G, 그리고 G를 누적하여 업데이트한 값이 Q라고 했을때

아래 그림에서 Q의 값은 어떻게 될까?

<img src='/assets/materials/20181118/incremental_mean.jpg' width=400px>

MC는 에피소드가 갱신될때마다, 해당 state action 페어가 가진 G값들의 평균을 내어 Q를 업데이트한다. 즉, 첫번째 episode의 G가 2인 상황에서 G값으로 8이 들어온다면, (2+8) /2로 Q의 값을 5로 업데이트한다.

과거의 G값을 계속 어딘가에 저장해두려면 귀찮고 번거롭다. 과거 G값없이도 현재의 에피소드 넘버와 직전 Q값, 그리고 현재 G값을 알면 $Q \leftarrow Q + {1\over N}(G-Q)$을 사용해 업데이트할 수 있다.

## Constant Alpha

그런데 $Q \leftarrow Q + {1 \over N}(G-Q)$는 한가지 문제가 있다.

G는 현재 에피소드의 return 값이다. 즉 최근의 관측값이다. 
Q는 직전까지의 G의 평균값으로, 우리가 믿고 가는 값이다.
G-Q는 그 둘 사이의 괴리로, 오차라고 생각할 수 있다.

위 식은 그 오차에다가 $1 \over N$만큼을 곱한 것을 Q에다 더해서 업데이트한다.
딥러닝에서 학습할때 쓰이는 learning rate $\eta$와 비슷한 느낌이다.

그런데 이 업데이트 폭이 $1 \over N$이라는 것은 초반 에피소드에서는 그 값이 크지만 나중에 가서는 엄청나게 작아짐을 의미한다.

후반부 에피소드에서 중요한 정보를 얻는다고 해도 그 정보의 가치가 전반부에 비해 훨씬 작아진다.

이러한 문제를 해결하기 위해서 $1 \over N$을 작은 상수 $\alpha$로 두는 방식이 제안되었다. 이를 Constant-alpha 방식이라고 한다.
$\alpha$를 크게 가져가면 업데이트 보폭이 커지므로 학습은 빨라지지만, 너무 큰 $\alpha$값은 optimal policy $\pi^*$로의 convergence를 저해할 수 있다.


```python
def get_probs(Q_s, epsilon, nA):
    non_greeey_prob = epsilon / nA
    policy_s = np.ones(nA) * non_greeey_prob
    
    greedy_idx = np.argmax(Q_s)
    policy_s[greedy_idx] += 1 - epsilon
    
    return policy_s
 
def generate_episode_from_Q(env, Q, epsilon, nA, max_episode_len):
    episode = []
    state = env.reset()
    cnt = 0
    while True:
        cnt += 1
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
            if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done or cnt == max_episode_len:
            break
    return episode


def update_Q(env, episode, Q, alpha, gamma):
    # Q <- Q + alpha(G-Q)

    states, actions, rewards = zip(*episode)
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
    for i, state in enumerate(states):
        action = actions[i]
        rewards_from_this_point = rewards[i:]
        corresponding_discount_rates = discounts[:-(1+i)]
        G = sum(rewards_from_this_point * corresponding_discount_rates)
        old_Q = Q[state][action]
        Q[state][action] = old_Q + alpha * (G - old_Q)
    return Q
    

def mc_control(env, num_episodes, alpha, gamma, eps_start=1.0, eps_min = 0.1, eps_decay_duration=3000):
    
    nA = env.action_space.n
    
    # Q table
    Q = defaultdict(lambda: np.zeros(nA))
    
    epsilon_decay_angle = (eps_min - eps_start) / eps_decay_duration
    epsilon = eps_start
    
    for i_episode in range(1, num_episodes + 1):
        
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        epsilon += epsilon_decay_angle
        epsilon = max(eps_min, epsilon)
        episode = generate_episode_from_Q(env, Q, epsilon, nA, max_episode_len=500)
        Q = update_Q(env, episode, Q, alpha, gamma)
        policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q

policy, Q = mc_control(env, 10000, 0.02, 0.99)
```

    Episode 10000/10000.


```python
visualize_policy(policy)
```


![png](/assets/materials/20181118/core_curriculum_1_56_0.png)


와... 신기하다!

## Temporal-Difference Methods

MC를 사용해 꽤 좋은 성과를 낼 수 있었다. 비록 절벽에 딱 붙어서 움직이는 효율적인 모습까지는 아니었지만 그래도 괜찮은 성능을 보였다.

MC 방식의 단점은 Episode가 끝난 후 업데이트가 이루어진다는 것이다. 위 구현에서 `max_episode_len` 파라미터를 500 정도로 설정해두었다. 왜 그랬을까?

`generate_episode_from_Q`에서는 action의 결과로서 game이 끝날때까지 (done이 True)일때까지 계속 action을 실행한다. Cliff Walking 문제에서 처음에 완전 랜덤한 액션을 하는 agent가 랜덤한 액션만으로 종착지까지 도달하도록 기대하는 것은 너무 어려운 문제다. 실제로 `max_episode_len`를 더 크게 설정하거나 아예 없애버리면 episode가 기약없이 계속 생성된다.

또 다른 문제는 continuous task에 대한 알고리즘의 적용이다. 시작과 끝이 정해져있지 않은 오픈 월드의 게임에서는 애초에 episode의 끝이 없기에 MC를 그대로 적용할 수 없다. MC에서는 episode가 끝나야 비로소 Q함수를 업데이트할 수 있기 때문이다.

Temporal-Difference Methods는 이러한 MC의 단점을 보완한다. 에피소드가 다 생성된 후에 Q함수를 업데이트하는 것이 아니라, TD는 매 타임스텝마다 value function을 업데이트한다.

## TD Control

TD 방법에는 우리가 많이 들어본 것들이 나온다.

### Sarsa(0) a.k.a Sarsa
Sarsa는 on-policy TD control method다.
Sarsa는 state, action, reward, state, action의 연속 시퀀스에서 따온 이름이다.

MC와 달리 TD는 매 타입스텝마다 Q함수를 업데이트한다고 했다.
즉, $G_t$를 구하는 것이 아닌, action의 결과로서 얻는 reward와 그 다음 state, action을 사용해서 $G_t$를 근사한다.

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))$

즉 매 타임스텝에서 s에 a를 써서 r, next_s를 얻는다.
그리고 현재 사용하는 $\epsilon-greedy$를 사용해 next_a를 뽑아놓는다.

그러면 현재 시점(t)에서 보자면, 무한히 먼 미래까지 확장된 $G_t$까지는 아니지만 그래도 다음 액션으로 인한 value까지는 감안한 value로 Q함수를 업데이트하게 되는 것이다.

Sarsa를 구현해보자.


```python
from collections import deque

def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    # Q[s, a] = Q[s, a] + a * (r_1 + gamma * Qsa_next - Qsa)
    Qsa = Qsa + alpha * (reward + gamma * Qsa_next - Qsa)
    return Qsa

def sarsa(env, num_episodes, alpha, gamma, eps_start=1.0, eps_min = 0.1, eps_decay_duration=3000):
    
    MAX_EPISODE_LENGTH = 300
    PLOT_EVERY = 100
    
    tmp_scores = deque(maxlen=PLOT_EVERY)
    scores = deque(maxlen=num_episodes)
    
    #init Q
    Q = defaultdict(lambda: np.zeros(env.nA))
    
    epsilon_decay_angle = (eps_min - eps_start) / eps_decay_duration
    epsilon = eps_start
    
    
    # loop over ep
    for i_episode in range(1, num_episodes+1):
        
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        score = 0
        
        epsilon += epsilon_decay_angle
        epsilon = max(eps_min, epsilon)
        
        # init state
        state = env.reset()
        
        policy = get_probs(Q[state], epsilon, env.nA)
        action = np.random.choice(np.arange(env.nA), p=policy)
        
        for i in range(MAX_EPISODE_LENGTH):
            
            next_state, reward, done, info = env.step(action)
            
            score += reward
            
            if done:
                
                Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                tmp_scores.append(score)
                break
            else:
                
                # next action
                policy = get_probs(Q[state], epsilon, env.nA)
                next_action = np.random.choice(np.arange(env.nA), p=policy)
                
                # update Q
                Q[state][action] = update_Q(Q[state][action], Q[next_state][next_action], reward, alpha, gamma)
                
                state = next_state
                action = next_action
                
        if i_episode % PLOT_EVERY == 0:
            scores.append(np.mean(tmp_scores))
            
    fig, ax = plt.subplots()
    
    ax.plot(np.linspace(0,num_episodes,len(scores),endpoint=False), np.array(scores))
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Average Reward (Over Next {:d} Episodes)'.format(PLOT_EVERY))
    plt.show()
    
    print("Best Average Reward over {:d} Episodes: {}".format(PLOT_EVERY, np.max(scores)))
    
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q
```


```python
policy, Q = sarsa(env, 10000, alpha=0.02, gamma=0.99)
```

    Episode 10000/10000


![png](/assets/materials/20181118/core_curriculum_1_61_1.png)


    Best Average Reward over 100 Episodes: -22.21



```python
visualize_policy(policy)
```


![png](/assets/materials/20181118/core_curriculum_1_62_0.png)


## Sarsamax

Sarsa는 학습은 잘 되었지만, 쫄보처럼 멀리 돌아서간다. 이번에는 Sarsamax로 해보자.
Sarsamax는 Q-learning이라고도 한다.

Sarsa는 s에서 a를 실행한 다음 얻는 next s에 현재 policy를 적용해 next a까지 뽑았다. 그리고 Qsa를 Qsa_next를 사용해서 업데이트했다.

Sarsamax는 s에서 a를 뽑은 다음 next s에서 next a를 또 뽑지 않는다. next s가 가질 수 있는 가장 높은 값을 사용해서 Qsa를 업데이트한다. 그러니까 next a를 뽑는 절차를 거치지 않는다.

$\epsilon-greedy$를 사용해서 a를 뽑았는데, 실제 Q함수를 업데이트할때 next state의 최댓값을 사용하므로 greed policy를 따른다고 볼 수 있다. 즉 실제 episode를 생성하는 policy와 학습에 사용하는 policy가 다르다. 이를 off-policy 방식이라고 한다.


```python
from collections import deque

def update_Q(Qsa, Qs_max, reward, alpha, gamma):
    # Q[s, a] = Q[s, a] + a * (r_1 + gamma * Qs_max - Qsa)
    Qsa = Qsa + alpha * (reward + gamma * Qs_max - Qsa)
    return Qsa

def sarsamax(env, num_episodes, alpha, gamma, eps_start=1.0, eps_min = 0.1, eps_decay_duration=3000):
    
    MAX_EPISODE_LENGTH = 300
    PLOT_EVERY = 100
    
    tmp_scores = deque(maxlen=PLOT_EVERY)
    scores = deque(maxlen=num_episodes)
    
    #init Q
    Q = defaultdict(lambda: np.zeros(env.nA))
    
    epsilon_decay_angle = (eps_min - eps_start) / eps_decay_duration
    epsilon = eps_start
    
    
    # loop over ep
    for i_episode in range(1, num_episodes+1):
        
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        score = 0
        
        epsilon += epsilon_decay_angle
        epsilon = max(eps_min, epsilon)
        
        # init state
        state = env.reset()
        
        policy = get_probs(Q[state], epsilon, env.nA)
        
        for i in range(MAX_EPISODE_LENGTH):
            
            #sarsamx에서는 action이 for loop 안으로 들어온다.
            action = np.random.choice(np.arange(env.nA), p=policy)
        
            next_state, reward, done, info = env.step(action)
            
            score += reward
            
            if done:
                
                Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                tmp_scores.append(score)
                break
            
            else:
                
                # update Q
                Q[state][action] = update_Q(Q[state][action], Q[next_state].max(), reward, alpha, gamma)
                
                state = next_state
                
        if i_episode % PLOT_EVERY == 0:
            scores.append(np.mean(tmp_scores))
            
    fig, ax = plt.subplots()
    
    ax.plot(np.linspace(0,num_episodes,len(scores),endpoint=False), np.array(scores))
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Average Reward (Over Next {:d} Episodes)'.format(PLOT_EVERY))
    plt.show()
    
    print("Best Average Reward over {:d} Episodes: {}".format(PLOT_EVERY, np.max(scores)))
    
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q
```


```python
policy, Q = sarsamax(env, 10000, alpha=0.02, gamma=0.99)
```

    Episode 100/10000

    /Users/junkwhinger/anaconda3/envs/pytorch/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /Users/junkwhinger/anaconda3/envs/pytorch/lib/python3.6/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)


    Episode 10000/10000


![png](/assets/materials/20181118/core_curriculum_1_65_3.png)


    Best Average Reward over 100 Episodes: nan



```python
visualize_policy(policy)
```


![png](/assets/materials/20181118/core_curriculum_1_66_0.png)


와씨 완벽하다!

### Expected Sarsa

Sarsamax가 next state의 value 중 최댓값을 골랐다면, expected sarsa는 policy를 활용해 기댓값을 구한다. 그때문에 expected라는 이름이 붙었다. 또 max값을 취하지 않기 때문에 policy를 그대로 사용하므로 on-policy method이다.

구현해보자.


```python
from collections import deque

def update_Q(Qsa, Qs_expected, reward, alpha, gamma):
    # Q[s, a] = Q[s, a] + a * (r_1 + gamma * Qs_max - Qsa)
    Qsa = Qsa + alpha * (reward + gamma * Qs_expected - Qsa)
    return Qsa

def expectedsarsa(env, num_episodes, alpha, gamma, eps_start=1.0, eps_min = 0.1, eps_decay_duration=3000):
    
    MAX_EPISODE_LENGTH = 300
    PLOT_EVERY = 100
    
    tmp_scores = deque(maxlen=PLOT_EVERY)
    scores = deque(maxlen=num_episodes)
    
    #init Q
    Q = defaultdict(lambda: np.zeros(env.nA))
    
    epsilon_decay_angle = (eps_min - eps_start) / eps_decay_duration
    epsilon = eps_start
    
    
    # loop over ep
    for i_episode in range(1, num_episodes+1):
        
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        score = 0
        
        epsilon += epsilon_decay_angle
        epsilon = max(eps_min, epsilon)
        
        # init state
        state = env.reset()
        
        policy = get_probs(Q[state], epsilon, env.nA)
        
        for i in range(MAX_EPISODE_LENGTH):
            
            action = np.random.choice(np.arange(env.nA), p=policy)
        
            next_state, reward, done, info = env.step(action)
            
            score += reward
            
            if done:
                
                Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                tmp_scores.append(score)
                break
            
            else:
                
                # update Q
                policy = get_probs(Q[next_state], epsilon, env.nA)
                expected_Qs = np.dot(policy, Q[next_state])
                Q[state][action] = update_Q(Q[state][action], expected_Qs, reward, alpha, gamma)
                
                state = next_state
                
        if i_episode % PLOT_EVERY == 0:
            scores.append(np.mean(tmp_scores))
            
    fig, ax = plt.subplots()
    
    ax.plot(np.linspace(0,num_episodes,len(scores),endpoint=False), np.array(scores))
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Average Reward (Over Next {:d} Episodes)'.format(PLOT_EVERY))
    plt.show()
    
    print("Best Average Reward over {:d} Episodes: {}".format(PLOT_EVERY, np.max(scores)))
    
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q
```


```python
policy, Q = expectedsarsa(env, 10000, alpha=0.02, gamma=0.99)
```

    Episode 10000/10000


![png](/assets/materials/20181118/core_curriculum_1_70_1.png)


    Best Average Reward over 100 Episodes: -18.79



```python
visualize_policy(policy)
```


![png](/assets/materials/20181118/core_curriculum_1_71_0.png)


expectedsarsa는 sarsa와 비슷한 결과를 보였다. 왜그랬을까?

sarsamax에서 어떤 state의 값은 그 다음 state의 max value가 반영되었다. 즉, 바로 옆에 절벽이 있어 큰 페널티를 받을 확률이 있더라도 greedy 판정에 의해 해당 위험은 무시된다.

하지만 expectedsarsa의 경우에는 그 확률도 감안한다. 위의 구현에서 epsilon은 0.1 이하로 떨어지지 않도록 고정되었다. 즉 아무리 작은 확률이라도 기댓값에 포함되기 때문에 이처럼 절벽으로부터 가급적 멀리 떨어져서 이동하려는 행태를 보이지 않나 싶다.


```python
state = env.reset()
for j in range(200):
    env.render(mode='human')
    sys.stdout.flush()
    action = policy[state]
    state, reward, done, _ = env.step(action)
    if done:
        env.render(mode='human')
        break 
        
env.close()
```

    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    x  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    x  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  o  o  o
    x  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    x  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  x  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  x  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  x  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  x  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  x  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  x  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  x  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  x  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  x  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  o  x  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  o  o  x
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  x
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  x
    o  C  C  C  C  C  C  C  C  C  C  T
    
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  o  o  o  o  o  o  o  o  o  o  o
    o  C  C  C  C  C  C  C  C  C  C  x
    


## Discrete vs. Continuous

지금까지 우리가 다룬 Cliff Walking 문제는 Discrete한 환경을 가지고 있다. env는 격자 형식의 grid world로 되어있다. agent의 이동은 셀 단위로 딱딱 나뉘어져 있다.

반대로 우리가 하고 싶은 자율주행이나 달리기, 로봇제어는 행동의 단위가 continuous하다.

그런데 지금까지 배운 알고리즘은 모두 discrete한 환경을 전제로 하니.. 어떡하지?

여러가지 방법이 있다.

하나는 continuous한 환경을 discrete하게 바꾸는 것이다. 이것이 바로 Discretization.

<img src="/assets/materials/20181118/discretization.png"></img>

격자의 크기를 Cliff-Walking처럼 균일(uniform)하게 가져가지 않고 어디는 조밀하게 어디는 밀도가 낮게 구성하는 방식을 Non-Uniform Discretization이라 한다.

Tile Coding이라는 방법도 있다. 다양한 크기의 격자를 가진 grid world를 여러개 생성한다. 그리고 해당 state를 처리할 때 각각의 grid world에 해당하는 영역을 대응시키는 방식이다.
<img src="/assets/materials/20181118/tile_coding.png"></img>


또 다른 방법으로는 Coarse Coding도 있다. Tile Coding과 비슷한데, 이는 해당 state를 더 sparse한 vector로 표현하는 방법이다. 예컨대 공간안에 원을 여러개 뿌려놓고 state s에 해당하는 원의 index에 속하면 1, 아니면 0을 주는 방식으로 state를 표현한다.

tile coding에서는 사각형 형태의 격자를 적용했지만 coarse coding에는 어떤 형태라도 상관없다. 만약 가우시안 분포를 따르는 형태로 coding을 하면, state의 위치가 특정 분포의 평균에 가깝다면 높은 값이 나오고 멀리간다면 낮은 값이 나온다. 즉, 1/0으로 처리되는 것이 아닌 continuous한 값으로 state를 정의할 수 있게 된다.

## Function Approximation

그런데 문제가 조금만 복잡해져도 필요한 discrete space가 엄청 커지기 때문에 이런 방식의 discretization은 적용하기 어려워진다. 또 근처에 위치한 state들은 값은 서로 비슷하거나 아니면 스무스하게 변하기 마련. discretization만으로는 이러한 특징을 잘 표현할 수 없다.

우리가 도달하려고 하는 이상향 $v_{pi}(s)$, $q_{pi}(s, a)$는 고차원 평면상의 continuous한 space임. 간단한 문제 몇몇을 제외하고는 이런 고차원 space의 특징을 완벽히 잡아내기란 실질적으로 불가능함. 그래서 이를 approximate(근사)하자는 거다. $v_{pi}(s)$, $q_{pi}(s, a)$는 각각 state-value function, action-value function이므로 우리가 하려는 것은 function approximation.

$\hat{v}(s) \approx v_{\pi}(s)$  
$\hat{q}(s, a) \approx q_{\pi}(s, a)$

근사를 하기 위해서는 어떤 변환작업을 거쳐야 하는데, 이때 사용할 parameter vector를 W라 하면 다음과 같이 표현할 수 있다.

$\hat{v}(s, W) \approx v_{\pi}(s)$  
$\hat{q}(s, a, W) \approx q_{\pi}(s, a)$

그리고 우리가 해야 할일은 어느정도 만족스런 근사를 할때까지 W를 업데이트시켜나가는 것임.

## How does it work?

가장 간단한 state-value function approximation을 살펴보자.
state s를 W에 통과시키면 $\hat{v}(s, W)$을 얻을 수 있다.
$\hat{v}(s, W)$에서 s는 state고, W는 parameter vector.
이를 통해서 얻는 결과는 하나의 스칼라값(value)이다.

이렇게 구하기 위해서 필요한 것은 state를 feature vector의 형식으로 표현해야 한다는 것. $X(s) = (x_1(s), x_2(s), ... , x_n(s))$

만약 state가 애초에 vector로 표현되어있다면 굳이 변환할 필요는 없지만, Cliff Walking 문제처럼 격자의 id로 되어있는 경우에는 변환이 필요하겠다.

feature vector화시키면서 다양한 변환을 적용할 수 있기 때문에 굳이 raw state만을 활용해야할 필요는 없다.

얻으려는 최종 결과값이 scalar로, state를 변환한 X(s)가 vector, W도 vector라면 어떻게 하지? dot product!

$\hat{v}(s, W) = X(s)^T \odot W$

dot product는 linear combination과 같으므로

$\hat{v}(s, W) = \Sigma_{j=1}^n x_j(s) w_j$

이런 방식을 linear function appoximation이라고 한다.
즉 underlying value function을 linear function을 사용해서 approximation하는 것임.

## Gradient Descent

$\hat{v}(s, W) = X(s)^T \odot W$은 linear function이므로,
$\triangledown_W \hat{v}(s, W) = X(s)$
w에 대한 $\hat{v}$의 derivative는 X(s)가 된다.

학습의 목표는 $\hat{v}$와 true value인 $v_{\pi}$와의 Error를 최소화하는 것.

즉, $J(w) = E_{\pi}[(v_{\pi}(s) - X(s)^T W)^2]$ 
RL의 도메인은 stochastic하므로 기댓값을 씌워준다.

이제 error의 w에 대한 gradient를 구한다.

$\triangledown_W J(W) = -2(v_{\pi}(s) - X(s)^TW)x(s)$
여기서는 E를 빼버렸다. 하나의 state s가 가리키는 error gradient에 집중하기 위해서.  s는 stochastic하게 결정된다. 샘플을 무한히 뽑다보면 결국 이 값도 평균에 수렴하게 된다.

이를 다음과 같은 식으로 업데이트한다.

$\triangle W = - \alpha {1\over2} \triangledown_{W} J(W)$  
$\triangle W = \alpha (v_{\pi}(s) - X(s)^T W)X(s)$

$\alpha$는 이미 아는 것처럼 step size, learning_rate parameter다.  
$-{1\over2}$는 derivative를 구할때 얻은 -2를 상쇄하는 상수다.
true function과 approximation function이 거의 같아질때까지 이 업데이트를 반복한다.

At every update, change weights ($\triangle W$) by step step $\alpha$ away from error ($(v_{\pi}(s) - X(s)^T W)$ in the direction of $X(s)$


### action-value function approximation

action-value function은 state s와 action a를 input으로 한다.

$\hat{q}(s, a, W) \approx q_{\pi}(s, a)$

state-value function에 action 인자가 하나 더 들어갈 뿐, $W$ parameter vector를 쓰는 것은 변하지 않는다.

다만 state s는 여러개의 a를 가지기 때문에 a의 갯수만큼 action-value function을 approximation하는 것은 비효율적으로 보인다.

또 자율주행을 한다고 했을때 핸들의 방향을 트는 action과 액셀을 밟은 action은 동시에 이루어지므로 개별 action-value function을 하나하나 계산하는 것보다, 모든 action에 대해 한번에 계산하는 것이 더 낫다.

즉, s에서의 action이 4가지라고 하면 s를 넣어 $\hat{q}(s, a_1, W)$부터 $\hat{q}(s, a_4, W)$까지의 value를 approximation한다.

이때 늘어난 action의 갯수에 대해서 가중치 계산을 해야 하므로,
곱해주는 W 파라미터 벡터를 벡터가 아닌 매트릭스로 정의한다.

action은 이때 vector가 되므로 이를 action-vector appoximation이라고 한다.

## Non-Linear Function Approximation

앞서 Linear Function을 사용해서 $\hat{q}(s, a, W)$을 정의하고, gradient descent 방식을 사용해서 업데이트하는 방식을 알아보았다.

그런데 Linear Function은 아무리 층을 깊게 쌓아도 결국 Linear Function으로 재정의할 수 있다. 그렇다는 것은 층의 깊이와 관계없이 표현할 수 있는 형태가 매우 제한적임을 의미한다.

고차원 space에 있는 복잡한 true value function을 approximation하기 위해서는 근사하는 function 역시 표현력이 좋아야 하며, 이를 위해 non-linear activation function $f$를 씌운다.

$\hat{v}(s, W) = f(X(s)^T \cdot W)$

음? 어디서 많이 본 수식이 나온다.
이게 바로 인공신경망의 기본적인 형태다.
