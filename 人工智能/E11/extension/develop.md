# RL note

## Therory  

### part 1
**actor**
- A Version0 
A = r1  

- A Version1
Reward delay  
A1 = G1 = r1 + r2 + ...  

- A Version2
r100 should not thank a1
A1 = G1 = x * r1 + x^2 * r2 + ...

- A version3
**Normalization**:
Gi - b(baseline) -> so that some is + some is -  
chess -> Heurisic 

- on policy
spend time: data is in for loop  
previous data is not neccesary good for now  
this is on-policy  

off-policy: learn from previous -> train faster
-> PPO: you need the difference between now and previous  

- randomness
you need to try all the action, the data is more different  

### part 2
**critic**  
given obs
give discounted cumulated reward  
train critic
--> MC monte-caelo  
    get some **finished** game  
    surprived  
--> TD(Temporal difference) approach
    only use: st at rt and st+1  
> obst action reward obst+1
Vst - gamma * Vst+1 = rt  

- version 3.5
At = Gt' - Vst  
Vst is a random action cumulated reward  
but this is one - average  

- version 4  
At = (rt + Vst+1) - Vst  
now is average - average  
Advantage Actor-Critic  
Tip:  
game graph -> CNN shared by C and A  

### part 3
Only Critic: **DQN**  
famous one -> rainbow  
video: 
> youtu.be/o_g9JUMw1Oc
> youtu.be/2-zGCx4iv_k  

**DQN**  
2018 Lee  
 
Vpi(s) = cumulated reward  
given actor pi & state(obs)  

2 method 
    MC -> regression   
    TD  Vst - Vst+1 = rt  
Compare:
    MC -> Ga can be very different
    TD -> Vst+1 can be inaccurate  
    same data can get different result  

Another Critic 
Qpi(s,a) : given state and action  

a **Conclution**:
pi's = argmax a Qpi(s, a)  
Vpi'(s) >= Vpi(s)  

- **tips**:
- [x] Target net work make it easier to train    

- [x] Epsilon Greedy -> random action
  epsilon would decay  
  Boltzmannn Exploration  use p(a|s)  

- [x] Replay Buffer  
  each exp is : (st, at, rt, st+1)   
  may from different policy --> off policy  
    pros: decrease the time ;batch is more diverse  
    cons: pi is different  

- [x] double DQN
  Q is over-estiimated  
  Q(st, at) <-> rt + max a Q'(st+1, a)  
  Q(st, at) <-> rt + Q'(st+1, argmax a Q(st+1, a))  
  Q' is taregt network  
  1 line of code  

- [x] dueling DQN
  Q(s, a) = A(s, a) + V(s)  
  change the network structure  
  change V cita can change the different a in the same s  
  constrain A(s, a)  

- [x] Prioritized Reply
  TD error is large -> good experience  

- [ ] ~~Multi-step~~
  both MC & TD  

- [x] Noisy Net
  episode begin --> add noise to Q(s, a)  
  openai & deepmind discover  

- [ ] ~~Distributional Q-function~~ 

## TODO
solve some question  
build a stong network  
then draw the graph and finish the report

openai https://github.com/gsurma/cartpole