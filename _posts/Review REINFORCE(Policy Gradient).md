# Review: REINFORCE(Policy Gradient)



## What are Policy Gradient Methods?

- Policy-Based Learning: search directly for the optimal policy
- Policy Gradient Learning: estimates the weights of an optimal policy through gradient ascent.
  - vs. Non-Gradient approaches
  - Goal is to find $\theta$ of policy network that maximise expected return 



## How does Policy Gradient work?

- reward is given at the end of episode (+1 for win, -1 for lose)
- Loop:
  - collect an episode
  - if WON, increase the probability of each (state, action) combination
  - if LOST, decrease the probability of each (state, action) combination



## Connections to Supervised Learning

- Common
  - Amend NN weights to increase the probability of the target label
    - PG: inputs (state image) - targets (action left)
    - SL: inputs (dog image) - targets ("dog")
- Differences
  - In SL, dataset does not change over time, but in RL dataset varies by episode.
    - In RL, collect episode, update NN weights, discard the epsiode and get a new one.
  - In RL, dataset can have multiple conflicting opinions.



## Problem Setup

- Trajectory $\tau$ = state-action sequence (w/o rewards)
  - ex) $s_0, a_0,s_1, a_1,s_2, a_2, ... , s_H, a_H, s_{H+1}$
  - No restriction on its length
  - $H$ stands for Horizon
  - Why use trajectory instead of episode?
    - cuz we can use trajectory for both episodic and continuing tasks.
    - In this case where reward is given only at the last state, trajectory correspond to the full episode.
- $R(\tau)$ = Reward from Trajectory $\tau$  
  - $R(\tau) = r_1 + r_2 + r_3 + ... + r_H + r_{H+1}$
- Goal is to find the weights $\theta$ that maximise expected return 
  - so that on average, the agent experiences trajectories with high return
- $U(\theta)$ = expected return
- $\max_{\theta} U(\theta)$ =  value of $\theta$ that maximises $U(\theta)$ 
- expected return is a weighted average of rewards from all possible trajectories
  - $U(\theta) = \Sigma_{\tau} P(\tau ; \theta) R(\tau)$
  - $P(\tau; \theta)$ = probability of trajectory $\tau$ when using $\theta$ for policy 



## REINFORCE

- $U(\theta) = \Sigma_{\tau} P(\tau ; \theta) R(\tau)​$
  - The goail is to find $\theta$ that maximises $U(\theta)$ 
- To maximize it we use Gradient Ascent.
  - Gradient Descent is designed to fine the **minimum** of a function
    - Gradient Ascent will find the **maximum**
  - Gradient Descent steps in the direction of the **negative direction**,
    - Gradient Ascent steps in the direction of the **gradient**
  - update step
    - $\theta \leftarrow \theta + \alpha \triangledown_{\theta} U(\theta)$
      - $\alpha$ = step size 
      - repeatedly apply this update step, in the hopes that $\theta$ converges to the value that maximises $U(\theta)$
    - difficult to get exact $\triangledown_{\theta}U(\theta)$  
      - Why? have to consider every possible trajectory
  - Sample
    - instead of calculating the gradient,
    - estimate the gradient $\triangledown_{\theta}U(\theta)$, we have to consider a few trajectories
  - Details
    - Use the policy $\pi_{\theta}$ to collect trajectories $\tau^{(1)}, \tau^{(2)}, ... \tau^{(m)}$
      -  $\tau^{(i)} = (s_0^{(i)}, a_0^{(i)}, s_1^{(i)}, a_1^{(i)}, ..., a_H^{(i)}, s_{H+1}^{(i)})$
    - Use the trajectories to estimate the gradient $\triangledown_{\theta}U(\theta)$
      - $\triangledown_{\theta}U(\theta) \approx \hat{g} = \frac{1}{m}\Sigma_{i=1}^m\Sigma_{t=0}^H \triangledown_{\theta}\log \pi_{\theta}(a_t^{(i)} | s_t^{(i)})R(\tau^{(i)})$ 
- Pseudocode for **REINFORCE**
  - 1. Use the policy $\pi_{\theta}$ to collect $m$ trajectories $\{\tau^{(1)}, \tau^{(2)}, ..., \tau^{(m)} \}$ with horizon $H$. We refer to the i-th trajectory as $ \tau^{(i)} = (s_0^{(i)}, a_0^{(i)}, ..., s_H^{(i)}, a_H^{(i)}, s_{H+1}^{(i)})$
    2. Use the trajectories to estimate the gradient $\triangledown_{\theta}U(\theta)​$
       $\triangledown_{\theta}U(\theta) \approx \hat{g} := \frac{1}{m}\Sigma_{i=1}^m\Sigma_{t=0}^H \triangledown_{\theta}\log\pi_{\theta}(a_t^{(i)} | s_t^{(i)})R(\tau^{(i)}) ​$
    3. Update the weights of the policy
       $\theta \leftarrow \theta + \alpha\hat{g}​$
    4. Loop over steps 1-3 



## Derivation

- how to derive the following
  - $\triangledown_{\theta}U(\theta) \approx \hat{g} := \frac{1}{m}\Sigma_{i=1}^m\Sigma_{t=0}^H \triangledown_{\theta}\log\pi_{\theta}(a_t^{(i)} | s_t^{(i)})R(\tau^{(i)}) ​$
  - $U(\theta) = \Sigma_{\tau} P(\tau ; \theta) R(\tau)​$
  - take derivative of both side..
    - $\triangledown_{\theta}U(\theta) = \triangledown_{\theta}\Sigma_{\tau} P(\tau ; \theta) R(\tau) $
  - put derivative inside Sigma
    - $= \Sigma_\tau \triangledown_{\theta}P(\tau;\theta)R(\tau)​$
  - Multiply $\frac{P(\tau; \theta)}{P(\tau; \theta)}​$
    - $= \Sigma_\tau \frac{P(\tau; \theta)}{P(\tau; \theta)} \triangledown_{\theta}P(\tau;\theta)R(\tau)$
  - Swap...
    - $= \Sigma_\tau P(\tau; \theta)\frac{\triangledown_{\theta}P(\tau;\theta)}{P(\tau; \theta)} R(\tau)​$
  - likelihood ratio trick (REINFORCE trick) $\triangledown_{\theta} \log(P(\tau; \theta)) = \frac{\triangledown_{\theta}P(\tau; \theta)}{P(\tau; \theta)}​$
    - $= \Sigma_\tau P(\tau; \theta) \triangledown_{\theta}\log(P(\tau; \theta)) R(\tau)$
  - Sample-based Estimate
    - instead of calculating all possible trajectories, we take $m​$ samples
    - $\triangledown_{\theta}U(\theta) \approx \hat{g} := \frac{1}{m}\Sigma_{i=1}^m\ \triangledown_{\theta} \log P(\tau; \theta)R(\tau^{(i)}) ​$
  - Finishing the Calculation
    - further simplify $\triangledown_{\theta} \log P(\tau; \theta)$
      - prob of trajectory is a chain of product of transition prob and action selection prob of each time step
        - $ = \triangledown_{\theta} \log \bigg[ \prod^H_{t=0} P(s^{(i)}_{t+1} | s^{(i)}_t, a^{(i)}_t)\pi_\theta(a^{(i)}_t|s^{(i)}_t) \bigg]​$
      - log(a*b) = log a + log b
        - $ = \triangledown_{\theta} \bigg[ \log  \Sigma^H_{t=0} P(s^{(i)}_{t+1} | s^{(i)}_t, a^{(i)}_t) + \Sigma^H_{t=0} \log \pi_\theta(a^{(i)}_t|s^{(i)}_t) \bigg]$
      - the gradient of the sum = the sum of gradients
        - $ = \triangledown_{\theta}  \log  \Sigma^H_{t=0} P(s^{(i)}_{t+1} | s^{(i)}_t, a^{(i)}_t) + \triangledown_{\theta}  \Sigma^H_{t=0} \log \pi_\theta(a^{(i)}_t|s^{(i)}_t)​$
      - first component has no dependence on $\theta$ so the gradient of it is $0$
        - $= \triangledown_{\theta}  \Sigma^H_{t=0} \log \pi_\theta(a^{(i)}_t|s^{(i)}_t)$
      - gradient of the sum as the sum of gradients
        - $= \Sigma^H_{t=0} \triangledown_{\theta} \log \pi_\theta (a^{(i)}_t | s^{(i)}_t) ​$ 
    - plug this in.. then we get
      - $\triangledown_{\theta}U(\theta) \approx \hat{g} := \frac{1}{m}\Sigma_{i=1}^m\ \Sigma^H_{t=0} \triangledown_{\theta} \log \pi_\theta (a^{(i)}_t | s^{(i)}_t) R(\tau^{(i)}) $

## Beyong REINFORCE

- main problems
  - inefficient update process. run the policy once, update once and throw away the trajectory
  - estimate $g$ is very noisy. The collected trajectory may not be representative of the policy.
  - no clear credit assignment. A trajectory may contain many good/bad actions and whether these actions are reinforced depends only on the final output.



## Noise Reduction

- One trajectory for computing gradient and policy update (for practical purposes)

  - the result of a sampled trajectory comes down to chance, and doesn't contain much information about policy

- easiest option: simply sample more trajectories using distributed computing

  - $g = \frac{1}{N}\Sigma_{i=1}^N \Sigma_{t=0}^H \log\pi_\theta(a_t^{(i)} | s_t^{(i)}) R_i$

- Rewards Normalization

  - benefit of distributed learning
  - in many cases, the distribution of rewards shifts as learning happens. Reward=1 might be really good in the beginning, but really had after 1000 training episode.
  - Learning can be improved if we normalize the rewards, where $\mu$ is the mean, and $\sigma$ the standard deviation.
  - Similar to Batch Normalization

   

## Credit Assignment

- $g = \Sigma_{t=0}^H(... + r_{t-1} + r_t + ...)\triangledown_\theta \log\pi_\theta(a_t | s_t)​$
- at time step $t$, even before an action is decided, the agent has already received all the rewards up until step $t-1$. 
- Because we have a Markov process, the action at time step $t$  can only affect the future reward.
- so $g = \Sigma_{t=0}^HR_t^{future} \triangledown_\theta \pi_\theta(a_t | s_t)​$
- wait, is it okay to change our gradient? what about our original goal of maximising the expected reward?
- It turns out that mathematically, ignoring past rewards might change the gradient for each specific trajectory, but it doens't change the averaged gradient. So even though the gradient is different during training, on average we are still maximizing the average reward. \



## Importance Sampling

- Limitations of REINFORCE
  - generate a trajectory, update policy with it, and throw it away.
- why throw away? 
  - We need to compute the gradient for the current policy with the trajectories that are representatitve of the current policy
- Using trajectories only once is too wasteful. Let's recycle!
  - How? by modifying trajectories to be representative of the new policy
  - => Importance Sampling
- Importance Sampling
  - We generate a trajectory $\tau$ using the policy $\pi_\theta$. Then the trajectory has a probability $P(\tau; \theta)$ to be sampled.
  - This trajectory $\tau$ can be "just by chance" sampled under the new policy with a different probability $P(\tau; \theta^\prime)$
  - Imagine we compute the average of some quantity $f(\tau)$. so.. $avg(f(\tau))$
  - Mathematically, this is equivalent to adding up all the $f(\tau)$ weighted by a new prob under new policy
    - $\Sigma_\tau P(\tau; \theta^\prime)f(\tau)$
    - We modify this as...
      - $ = \Sigma_\tau \textcolor{red}{\frac{P(\tau; \theta)}{P(\tau; \theta)}} P(\tau; \theta^\prime)f(\tau)$
      - $ = \Sigma_{\tau} \textcolor{red}{P(\tau; \theta)} \frac{P(\tau; \theta^\prime)}{\textcolor{red}{P(\tau; \theta)}}f(\tau)$
      - First part: coefficient for sampling under the old policy,
      - second part: re-weighting factor
    - Intuition: this tells us that we can use old trajectories for computing averages for new policy, as long as we add extra re-weighting factor, that takes into account how under or over-representted each trajectory is under the new policy compared to the old one.
- The re-weighting factor
  - re-weighting factor = $\frac{P(\tau; \theta^\prime)}{P(\tau; \theta)}​$
  - each prob is multiplication of action probs of all steps under new or old policy
    - $\frac{P(\tau; \theta^\prime)}{P(\tau; \theta)} = \frac{\pi_{\theta^\prime}(a_1|s_1)\pi_{\theta^\prime}(a_2|s_2)\pi_{\theta^\prime}(a_3|s_3)...}{\pi_{\theta}(a_1|s_1)\pi_{\theta}(a_2|s_2)\pi_{\theta}(a_3|s_3)...}$
  - Problems
    - 1. this formula looks too complicated
      2. bigger problem: when some of policy gets close to zero, the re-weighting factor can become close to zero or 1 over 0 which diverges to infinity
    - when this happens, the re-weighting trick becomes unreliable. So in practice, we want to make sure the re-weighting factor is not too far from 1 when we utilize importance sampling.



## PPO: The Surrogate Function

- Re-weighting the Policy Gradient
  - Suppose we are trying to update our current policy $\pi_{\theta^\prime}$. 
    - To do that we need to estimate a gradient $g$.
    - But we only have trajectories generated by an older policy $\pi_\theta$
    - We need trajectories under our current policy to compute $g​$!
  - To use old trajectories we use importance sampling (re-weighting factor)
    - $g = \frac{P(\tau; \theta^\prime)}{P(\tau; \theta)} \Sigma_t \frac{\triangledown_{\theta^\prime} \pi_{\theta^\prime}(a_t|s_t)}{\pi_{\theta^\prime}(a_t|s_t)} R^{future}_t ​$
    - put the reweighting factor inside $\Sigma$
      - $g = \Sigma_t\frac{P(\tau; \theta^\prime)}{P(\tau; \theta)}  \frac{\triangledown_{\theta^\prime} \pi_{\theta^\prime}(a_t|s_t)}{\pi_{\theta^\prime}(a_t|s_t)} R^{future}_t ​$
    - the re-weighting factor is just the product of all the policy across each step
      - $g = \Sigma_t\frac{... \pi_{\theta^\prime}(a_t|s_t) ...}{... \pi_{\theta}(a_t|s_t) ...}  \frac{\triangledown_{\theta^\prime} \pi_{\theta^\prime}(a_t|s_t)}{\pi_{\theta^\prime}(a_t|s_t)} R^{future}_t ​$
    - Then we can cancel some terms
      - $g = \Sigma_t\frac{...\cancel{ \pi_{\theta^\prime}(a_t|s_t)} ...}{... \pi_{\theta}(a_t|s_t) ...}  \frac{\triangledown_{\theta^\prime} \pi_{\theta^\prime}(a_t|s_t)} {\cancel{\pi_{\theta^\prime}(a_t|s_t)}} R^{future}_t $
      - but what about the rest...?
    - Proximal Policy comes in.. if the old and current policy is close enough to each other, all the factors inside the "..." would be pretty close to 1. Then we can ignore them.
      - $g = \Sigma_t \frac{\triangledown_{\theta^\prime}\pi_{\theta^\prime}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}R^{future}_t$
      - this looks very similar to the old policy gradient
        - $g = \Sigma_{t} \frac{\triangledown_\theta \pi_\theta(a_t | s_t)}{\pi_{\theta}(a_t|s_t)} R_t^{future} $
        - where numer and demon use the same policy.
        - PPO is different because we are using two different policies
- The Surrogate Function
  - now we have the approximate form of the gradient
    - $g = \triangledown_{\theta^\prime} L_{sur}(\theta^\prime, \theta)$
  - we can think of it as the gradient of a new object, called the surrogate function
    - $L_{sur}(\theta^\prime, \theta) = \Sigma_t \frac{\pi_{\theta^\prime}(a_t |s_t)}{\pi_{\theta}(a_t|s_t)} R^{future}_t$
  - using this new gradient, we can perform gradient ascent to update our policy -- which can be thought as directly maximize the surrogate function
- still one important issue..
  - if we keep reusing old trajectories and updating our policy, at some point the new policy might become different enough from the old one, so that all the approximations we made could become invalid.
  - Need to make sure this doesn't happen.



## PPO: Clipping Policy Updates

- The Policy / Reward Cliff
  - What's the problem with updating our policy and ignoring the fact that the approximations are not valid anymore? One problem is it could lead to a really bad policy that is very hard to recover from.
  - $L_sur$ approximates the reward pretty well around the current policy, but it diverges from the actual reward when it's far away from the current policy.
  - So even if the policy is really bad and actual reward is low, the surrogate function would keep telling that the average reward is really good.
  - And if the policy is now stuck in a deep and flat bottom, the future updates won't be able to bring the policy back up!
  - How to fix this? 
- Clipped Surrogate Function
  - Idea: what if we just flatten the surrogate function? 
    - So if the average reward gets too high, then set the gradient to 0.
  - $L^{clip}_{sur}(\theta^\prime, \theta) = \Sigma_t \min \bigg\{ \frac{\pi_{\theta^\prime}(a_t|s_t)}{\pi_\theta (a_t|s_t)} R^{future}_t, clip_{\epsilon} \bigg(\frac{\pi_{\theta^\prime}(a_t|s_t)}{\pi_\theta (a_t|s_t)}   \bigg) R^{future}_t \bigg\}$
  - We want to make sure the two policy is similar, or that the ratio is close to 1. So we choose a small $\epsilon$ (typically 0.1 or 0.2), and apply the *clip* function to force the ratio to be within the interval $[1-\epsilon, 1+\epsilon]$
  - Now the ratio is clipped in two places. But we only want to clip the top part and not the bottom part. Why? Gradient Ascent
  - 