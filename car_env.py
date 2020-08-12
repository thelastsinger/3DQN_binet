#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
#sumo环境

tools_path = 'D:/ChromeCoreDownloads/sumo-win64-0.32.0/sumo-0.32.0/tools'
sumoBinary = "D:/ChromeCoreDownloads/sumo-win64-0.32.0/sumo-0.32.0/bin/sumo"
sumoConfig = "D:/Desktop/sumoconfignew/3Dnew.sumocfg"
#SUMO环境代码
import os, sys
sys.path.append(tools_path)
sys.path.append(os.path.join('D:/ChromeCoreDownloads/sumo-win64-0.32.0/sumo-0.32.0/tools'))
import traci
import traci.constants as tc
import numpy as np
import datetime

baseImpatience = 0.01 #基础不耐烦指数
timeToMaxImpatience = 300 #达到最大不难烦阀值

outputfile = open('output/queue_length.csv','w+')

print("phase_1,max_1,sum_1,phase_2,max_2,sum_2,phase_3,max_3,sum_3,phase_4,max_4,sum_4",file = outputfile)

#Environment Model
sumoCmd = [sumoBinary, "-c", sumoConfig]  #The path to the sumo.cfg file

#最大绿灯时间和最小绿灯时间
Gmin = 15
Gmax = 60

#奖励函数的权重分配
w1 = 0 / 10
w2 =1 # 10 / 10
w3 = 0 / 10
w4 = 0

#重置环境,并返回场景中所有交通灯的ID列表
def reset():
    traci.start(sumoCmd)
    tls = traci.trafficlights.getIDList()
    return tls

#获取状态
# def getState():   #原地图600m长5车道地图获取状态，获取状态只考虑进口且只考虑路口前150m状态
#     for veh_id in traci.vehicle.getIDList():
#         traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCUMULATED_WAITING_TIME))
#     p_state_2 = np.zeros((20,30,2))
#     for veh_id in traci.vehicle.getIDList():
#         p = traci.vehicle.getSubscriptionResults(veh_id) #获取订阅值（即获取对应车辆的位置和速度值）
#         ps = p[tc.VAR_POSITION]
#         spd = p[tc.VAR_SPEED]
#     #方法二:拼接
#         if (ps[0] > 431 and ps[0] < 581 and ps[1] > 586 and ps[1] < 599): #西进口
#             p_state_2[int((ps[1]-586)/3.3), int((ps[0]-431)/5)] = [1, round(spd/13.89)]
#         elif (ps[0] > 625 and ps[0] < 775 and ps[1] > 598 and ps[1] < 611): #东进口
#             p_state_2[int((ps[1]-598)/3.3)+4, int((ps[0]-625)/5)] = [1, round(spd/13.89)]
#         elif (ps[0] > 584 and ps[0] < 603 and ps[1] > 614 and ps[1] < 764): #北进口
#             p_state_2[int((ps[0]-584)/3.3)+8, int((ps[1]-614)/5)] = [1, round(spd/13.89)]
#         elif (ps[0] > 603 and ps[0] < 622 and ps[1] > 433 and ps[1] < 583): #南进口
#             p_state_2[int((ps[0]-603)/3.3)+14, int((ps[1]-433)/5)] = [1, round(spd/13.89)]
#     p_state_2 = np.reshape(p_state_2, [-1, 600, 2])
#     return p_state_2


# def getState():#自己写的非精确的状态的获取，状态为10m路段的车辆密度和平均速度（另加转向信息）
#     for veh_id in traci.vehicle.getIDList():
#         traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCUMULATED_WAITING_TIME))
#     p_state_2 = np.zeros((4,15,2))
#     for veh_id in traci.vehicle.getIDList():
#         p = traci.vehicle.getSubscriptionResults(veh_id) #获取订阅值（即获取对应车辆的位置和速度值）
#         ps = p[tc.VAR_POSITION]
#         spd = p[tc.VAR_SPEED]
#         # s = ""
#         # veh_signal = traci.vehicle.getSignals(veh_id)
#         # if(veh_signal & 1 > 0):
#         #     s = "right_turn"
#         # if(veh_signal & 2 > 0):
#         #     s = "left_turn"
#         # print(veh_id,s,"\n")
#         if (ps[0] > 0 and ps[0] < 150 and ps[1] > -10 and ps[1] < 0):  # 车辆处于西进口
#             p_state_2[0, int((ps[0] - 0) / 10), 0] += 1  # 对应块的车辆数量+1
#             p_state_2[0, int((ps[0] - 0) / 10), 1] += spd / 13.89  # 对应块的车速加上此车辆的车速除以最大车速
#         if (ps[0] > 150 and ps[0] < 300 and ps[1] > 0 and ps[1] < 10):  # 车辆位于东进口
#             p_state_2[1, int(abs(ps[0] - 300) / 10), 0] += 1
#             p_state_2[1, int(abs(ps[0] - 300) / 10), 1] += spd / 13.89
#         if (ps[1] > 0 and ps[1] < 150 and ps[0] > 140 and ps[0] < 150):  # 车辆处于北进口
#             p_state_2[2, int(abs(ps[1] - 150) / 10), 0] += 1
#             p_state_2[2, int(abs(ps[1] - 150) / 10), 1] += spd / 13.89
#         if (ps[1] > -150 and ps[1] < 0 and ps[0] > 150 and ps[0] < 160):  # 车辆处于南进口
#             p_state_2[3, int(abs(ps[1] + 150) / 10), 0] += 1
#             p_state_2[3, int(abs(ps[1] + 150) / 10), 1] += spd / 13.89
#     p_state_2 = p_state_2/6   #最多容纳6辆车，算是计算密度
#     p_state_2 = np.reshape(p_state_2, [-1, 60, 2])
#
#     return p_state_2


# def getState():  # 带转向灯信息的非精确状态
#     for veh_id in traci.vehicle.getIDList():
#         traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCUMULATED_WAITING_TIME))
#     p_state_2 = np.zeros((4,15,4))
#     for veh_id in traci.vehicle.getIDList():
#         p = traci.vehicle.getSubscriptionResults(veh_id) #获取订阅值（即获取对应车辆的位置和速度值）
#         ps = p[tc.VAR_POSITION]
#         spd = p[tc.VAR_SPEED]
#         # s = ""
#         veh_signal = traci.vehicle.getSignals(veh_id)
#         if (ps[0] > 0 and ps[0] < 150 and ps[1] > -10 and ps[1] < 0):  # 车辆处于西进口
#             p_state_2[0, int((ps[0] - 0) / 10), 0] += 1  # 对应块的车辆数量+1
#             p_state_2[0, int((ps[0] - 0) / 10), 1] += spd / 13.89  # 对应块的车速加上此车辆的车速除以最大车速
#             if (veh_signal & 1 > 0):
#                 p_state_2[0, int((ps[0] - 0) / 10), 2] += 1 # 对应块的右转向车辆数+1
#             if (veh_signal & 2 > 0):
#                 p_state_2[0, int((ps[0] - 0) / 10), 3] += 1 # 对应块的左转向车辆数+1
#         if (ps[0] > 150 and ps[0] < 300 and ps[1] > 0 and ps[1] < 10):  # 车辆位于东进口
#             p_state_2[1, int(abs(ps[0] - 300) / 10), 0] += 1
#             p_state_2[1, int(abs(ps[0] - 300) / 10), 1] += spd / 13.89
#             if (veh_signal & 1 > 0):
#                 p_state_2[1, int(abs(ps[0] - 300) / 10), 2] += 1 # 对应块的右转向车辆数+1
#             if (veh_signal & 2 > 0):
#                 p_state_2[1, int(abs(ps[0] - 300) / 10), 3] += 1 # 对应块的左转向车辆数+1
#         if (ps[1] > 0 and ps[1] < 150 and ps[0] > 140 and ps[0] < 150):  # 车辆处于北进口
#             p_state_2[2, int(abs(ps[1] - 150) / 10), 0] += 1
#             p_state_2[2, int(abs(ps[1] - 150) / 10), 1] += spd / 13.89
#             if (veh_signal & 1 > 0):
#                 p_state_2[2, int(abs(ps[1] - 150) / 10), 2] += 1 # 对应块的右转向车辆数+1
#             if (veh_signal & 2 > 0):
#                 p_state_2[2, int(abs(ps[1] - 150) / 10), 3] += 1 # 对应块的左转向车辆数+1
#         if (ps[1] > -150 and ps[1] < 0 and ps[0] > 150 and ps[0] < 160):  # 车辆处于南进口
#             p_state_2[3, int(abs(ps[1] + 150) / 10), 0] += 1
#             p_state_2[3, int(abs(ps[1] + 150) / 10), 1] += spd / 13.89
#             if (veh_signal & 1 > 0):
#                 p_state_2[3, int(abs(ps[1] + 150) / 10), 2] += 1 # 对应块的右转向车辆数+1
#             if (veh_signal & 2 > 0):
#                 p_state_2[3, int(abs(ps[1] + 150) / 10), 3] += 1 # 对应块的左转向车辆数+1
#     p_state_2 = p_state_2/6
#     p_state_2 = np.reshape(p_state_2, [-1, 60, 4])
#
#     return p_state_2



# def getState():  #对应3dqn里的状态计算方法，xy都除以5，有相当多的信息浪费
#     for veh_id in traci.vehicle.getIDList():
#         traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
#     p = traci.vehicle.getSubscriptionResults()
#     p_state = np.zeros((60, 60, 2))
#     for x in p:
#         ps = p[x][tc.VAR_POSITION]
#         spd = p[x][tc.VAR_SPEED]
#         p_state[int(ps[0] / 5), int(ps[1] / 5)] = [1, int(round(spd))]#一个位置矩阵一个速度矩阵
#     #         v_state[int(ps[0]/5), int(ps[1]/5)] = spd
#     p_state = np.reshape(p_state, [-1, 3600, 2])
#     return p_state  # , v_state]


#------------------------------------------------------------------------------------------
# 大地图获取状态
# def getState():  # 对应3dqn里的状态计算方法，xy都除以5，有相当多的信息浪费
#     for veh_id in traci.vehicle.getIDList():
#         traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
#     p = traci.vehicle.getSubscriptionResults()
#     p_state = np.zeros((120, 120, 2))
#     for x in p:
#         ps = p[x][tc.VAR_POSITION]
#         spd = p[x][tc.VAR_SPEED]
#         p_state[int(ps[0] / 5), int(ps[1] / 5)] = [1, int(round(spd))]#一个位置矩阵一个速度矩阵
#     #         v_state[int(ps[0]/5), int(ps[1]/5)] = spd
#     p_state = np.reshape(p_state, [-1, 14400, 2])
#     return p_state  # , v_state]

def getState():  # 带转向灯信息的非精确状态
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCUMULATED_WAITING_TIME))
    p_state_2 = np.zeros((4,30,4))
    for veh_id in traci.vehicle.getIDList():
        p = traci.vehicle.getSubscriptionResults(veh_id) #获取订阅值（即获取对应车辆的位置和速度值）
        ps = p[tc.VAR_POSITION]
        spd = p[tc.VAR_SPEED]
        # s = ""
        veh_signal = traci.vehicle.getSignals(veh_id)
        if (ps[0] > 0 and ps[0] < 300 and ps[1] > -14 and ps[1] < 0):  # 车辆处于西进口
            p_state_2[0, int((ps[0] - 0) / 10), 0] += 1  # 对应块的车辆数量+1
            p_state_2[0, int((ps[0] - 0) / 10), 1] += spd / 13.89  # 对应块的车速加上此车辆的车速除以最大车速
            if (veh_signal & 1 > 0):
                p_state_2[0, int((ps[0] - 0) / 10), 2] += 1 # 对应块的右转向车辆数+1
            if (veh_signal & 2 > 0):
                p_state_2[0, int((ps[0] - 0) / 10), 3] += 1 # 对应块的左转向车辆数+1
        if (ps[0] > 300 and ps[0] < 600 and ps[1] > 0 and ps[1] < 14):  # 车辆位于东进口
            p_state_2[1, int(abs(ps[0] - 600) / 10), 0] += 1
            p_state_2[1, int(abs(ps[0] - 600) / 10), 1] += spd / 13.89
            if (veh_signal & 1 > 0):
                p_state_2[1, int(abs(ps[0] - 600) / 10), 2] += 1 # 对应块的右转向车辆数+1
            if (veh_signal & 2 > 0):
                p_state_2[1, int(abs(ps[0] - 600) / 10), 3] += 1 # 对应块的左转向车辆数+1
        if (ps[1] > 0 and ps[1] < 300 and ps[0] > 286 and ps[0] < 300):  # 车辆处于北进口
            p_state_2[2, int(abs(ps[1] - 300) / 10), 0] += 1
            p_state_2[2, int(abs(ps[1] - 300) / 10), 1] += spd / 13.89
            if (veh_signal & 1 > 0):
                p_state_2[2, int(abs(ps[1] - 300) / 10), 2] += 1 # 对应块的右转向车辆数+1
            if (veh_signal & 2 > 0):
                p_state_2[2, int(abs(ps[1] - 300) / 10), 3] += 1 # 对应块的左转向车辆数+1
        if (ps[1] > -300 and ps[1] < 0 and ps[0] > 300 and ps[0] < 314):  # 车辆处于南进口
            p_state_2[3, int(abs(ps[1] + 300) / 10), 0] += 1
            p_state_2[3, int(abs(ps[1] + 300) / 10), 1] += spd / 13.89
            if (veh_signal & 1 > 0):
                p_state_2[3, int(abs(ps[1] + 300) / 10), 2] += 1 # 对应块的右转向车辆数+1
            if (veh_signal & 2 > 0):
                p_state_2[3, int(abs(ps[1] + 300) / 10), 3] += 1 # 对应块的左转向车辆数+1
    p_state_2 = p_state_2/8
    p_state_2 = np.reshape(p_state_2, [-1, 120, 4])

    return p_state_2



#------------------------------------------------------------------------------------------

# 保证信号配时方案在合理范围内
def getCorrectCycle(phases):
    for i in range(len(phases)):
        if phases[i] < Gmin:
            phases[i] = Gmin
        if phases[i] > Gmax:
            phases[i] = Gmax
    return phases

# 交通灯在当前的周期信号配时状态下,可以选择的合法动作集合,将相位绿灯时间转换成0~8这样的数,另外-1也要特殊处理
def getLegalAction(phases):
    # 根据当前的相位，如果不能减5或加5相应的action处为-1
    legal_action = np.zeros(9)-1
    i = 0
    for x in phases: # phases是一个长度为4的list,初始化值
        if x - 5 > Gmin:
            legal_action[i] = i
        if x + 5 <= Gmax:
            legal_action[i+5] = i+5
        i += 1
    legal_action[4] = 4
    return legal_action# 这个循环的意思是如果四个相位的时间都在5-60间，legal_action就是[0,1,2...7,8]这样一个序列，如果有一个不在区间内，那么对应的i和i+5的值就是-1(既不能执行该动作)

#根据当前的相位和选择的动作调整得到新的相位配时方案
# def getPhaseFromAction(phases, act):
#     if act<4: #如果选择的动作序号小于4，那对应的动作相位时间-5s
#         phases[int(act)] -= 5
#     elif act>4: #如果选择的动作序号大于4，那么对应的动作相位时间+5s
#         phases[int(act)-5] += 5
#     return phases
def getPhaseFromAction(phases, act, ns):   #新动作,如果ns为1则是对ns绿灯相位进行修改，为0则对we相位修改
    if ns == 1:
        if act == 0:
            phases[0] += 5
    return phases

#the process of the action
#input: traffic light; new phases; waiting time in the beginning of this cycle
#output: new state; reward; End or not(Bool); new waiting time at the end of the next cycle

def step(tls, ph, wait_time_map, pre_queue, pre_delay, pre_throughput, pre_impatience): #一个step相当于一个周期
    # 一个action相当于一次s，a，r，s1的过程
    tls_id = tls[0]                    # 获取当前交通灯的id，因为单交叉口只有一个所以取0,多路口的话可能要遍历
    init_p = traci.trafficlights.getPhase(tls_id) # 获取交通灯的当前处于第几相位,返回索引
    prev = -1
    changed = False
    queue_max = []   # 存放一周期内的所有相位的排队长度
    queue_sum = []
    indx = []
    dets = traci.lanearea.getIDList() # dector获取队列信息

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:              # 路网还有车则继续
        c_p = traci.trafficlights.getPhase(tls_id)                  # 获取交通灯的当前相位情况,四相位的话返回0,1,2,3,4,5,6,7
        if c_p != prev and c_p%2 == 0:                              # 如果当前相位序号不是前一个相位且为序号偶数
            if step > ph[0]:                                        # 当第一个相位运行结束之后
                queue_length=[]
                for det in dets:
                    queue=traci.lanearea.getJamLengthVehicle(det)
                    queue_length.append(queue)
                queue_max.append( max(queue_length))
                queue_sum.append(sum(queue_length))  # /len(queue_length)
                indx.append(np.argmax(queue_length))
            traci.trafficlights.setPhaseDuration(tls_id, ph[int(c_p/2)]-0.5)
            # 设置当前相位的持续时间为相位序号int(c_p/2)对应时长再-0.5后的时间（暂时不知道意义）
            prev = c_p                   # prev表示前一个相位
        if init_p != c_p:                # 如果初始相位与当前相位不同，表示相位已经改变
            changed = True
        if changed:
            if c_p == init_p:            # 如果当前相位等于初始相位，表示一个周期结束
                break
        traci.simulationStep()           # 向前模拟1s
        step += 1
        # 车辆的累计等待时间
        if step % 5 == 0:
            for veh_id in traci.vehicle.getIDList():
                wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    p_state = getState()
    # print(str(indx[0]) + "," + str(queue_max[0]) + "," + str(queue_sum[0]) + "," + str(indx[1]) + "," + str(
    #     queue_max[1]) + "," + str(queue_sum[1]) +
    #       "," + str(indx[2]) + "," + str(queue_max[2]) + "," + str(queue_sum[2]) + "," + str(indx[3]) + "," + str(
    #     queue_max[3]) + "," + str(queue_sum[3]), file=outputfile)

    wait_t = dict(wait_time_map)                           # 计算所有车辆的累计等待时间的和
    impatience = 0.0                                      # 驾驶员不耐烦程度初始值
    for i in wait_t:                                      # 计算驾驶员不耐烦程度的总和
        impatience += max(0.0, min(1.0, baseImpatience + wait_t[i] / timeToMaxImpatience))

    done = False
    if traci.simulation.getMinExpectedNumber() == 0:    # 判断路网中的车辆是否为空,为空则标记为结束，结束的标准可以再加
        done = True                                     # 结束的标准可以修改,这个在高峰期不太合适,可以改成指定时段内
    # print('四相位最长车道序号和排队车辆数分别为', indx, queue_max)  # 输出每个路段的最大排队长度和车道
    # 奖励函数的设计部分



    # 1.最长的排队长度
    max_queue = max(queue_max)
    reward1 = pre_queue**2 - max_queue**2 
    # 2.四相位结束之后,累计等待时延的差值
    current_delay = sum(wait_t[x] for x in wait_t)
    reward2 = pre_delay - current_delay
    # 3.四相位结束之后,累积的吞吐量
    current_throughput = len(wait_t) 
    reward3 = current_throughput
    # 4.驾驶员的不耐烦程度 MAX(0, MIN(1.0, baseImpatience + waitingTime / timeToMaxImpatience))   ----sumo中不耐烦值的计算方式
    reward4 = pre_impatience - impatience  # 上一个周期的不耐烦程度减当前周期为奖励，若不耐烦程度减少，则奖励会增加

    reward = w1 * reward1 + w2 * reward2 + w3 * reward3 + w4 * reward4




    q_m = sum(queue_sum)           # 平均排队长度
    return p_state, reward, done, wait_t, q_m, max_queue, current_delay, current_throughput, impatience       # 返回状态矩阵，奖励值，是否最终状态，本次的累计等待总时间

# 一个episode后关闭sumo
def end():
    traci.close()
