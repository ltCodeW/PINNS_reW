# 使用整个流场的数据，并且量纲化
# 只是用了xyz的坐标
# 按照PINNs的思路，使用自己的数据

# 目前结果

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import time
np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    def __init__(self, X,Y,Z,NQ,U,V,W,P,Rho,onWall,layers):
        # comblinedX = np.concatenate([X,Y,Z,a,H,Ma,r,A], 1)
        comblinedX = np.concatenate([X,Y,Z], 1)

        self.lb = comblinedX.min(0)
        self.ub = comblinedX.max(0)

        self.X = X
        self.Y = Y
        self.Z = Z
        self.NQ = NQ
        self.U = U
        self.V = V
        self.W = W
        self.P = P
        self.Rho = Rho
        self.onWall = onWall
        self.layers = layers

        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.X_tf = tf.placeholder(tf.float32, shape=[None, self.X.shape[1]])
        self.Y_tf = tf.placeholder(tf.float32, shape=[None, self.Y.shape[1]])
        self.Z_tf = tf.placeholder(tf.float32, shape=[None, self.Z.shape[1]])
        self.Rho_tf = tf.placeholder(tf.float32, shape=[None, self.Rho.shape[1]])
        self.onWall_tf = tf.placeholder(tf.float32, shape=[None, self.onWall.shape[1]])
        self.NQ_tf = tf.placeholder(tf.float32, shape=[None, self.NQ.shape[1]])
        self.U_tf = tf.placeholder(tf.float32, shape=[None, self.U.shape[1]])
        self.V_tf = tf.placeholder(tf.float32, shape=[None, self.V.shape[1]])
        self.W_tf = tf.placeholder(tf.float32, shape=[None, self.W.shape[1]])
        self.P_tf = tf.placeholder(tf.float32, shape=[None, self.P.shape[1]])


        self.U_pred, self.V_pred, self.W_pred, \
        self.P_pred, self.Rho_pred,self.NQ_pred,\
        self.f_1, self.f_2, self.f_3, self.f_4,self.f_5= self.net_NS(self.X_tf, self.Y_tf,self.Z_tf,
                                    self.onWall_tf )



        self.lu =tf.reduce_sum(tf.square(self.U_tf - self.U_pred))
        self.lv = tf.reduce_sum(tf.square(self.V_tf - self.V_pred))
        self.lw = tf.reduce_sum(tf.square(self.W_tf - self.W_pred))
        self.lp = tf.reduce_sum(tf.square(self.P_tf - self.P_pred))
        self.lrho = tf.reduce_sum(tf.square(self.Rho_tf - self.Rho_pred))
        self.lq = tf.reduce_sum(tf.square(self.NQ_tf - self.NQ_pred))
        self.l1 = tf.reduce_sum(tf.square(self.f_1))
        self.l2 = tf.reduce_sum(tf.square(self.f_2))
        self.l3 = tf.reduce_sum(tf.square(self.f_3))
        self.l4 =tf.reduce_sum(tf.square(self.f_4))
        self.l5 =tf.reduce_sum(tf.square(self.f_5))

        self.loss = self.lu+ self.lv+ self.lw +self.lp + self.lrho\
                    + self.lq\
                    +self.l1+self.l2+self.l3+self.l4+self.l5


        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases


    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def getval(self, preVal):
        psi = preVal[:, 0:1]
        P = preVal[:, 1:2]
        Rho = preVal[:, 2:3]
        return psi, P, Rho


    def net_NS(self,X,Y,Z,onWall):
        preVal= self.neural_net(tf.concat([X,Y,Z],1), self.weights, self.biases)
        psi,P, Rho = self.getval(preVal)
        U = tf.gradients(psi, X)[0]  # psi关于y的导数
        V = tf.gradients(psi, Y)[0]
        W = tf.gradients(psi, Z)[0]

        gamma = 1.4
        Pr = 0.72
        # 在读取得文件中Re为9.2259328E+03，但是没有无量纲化，这里计算无量纲化数据
        # T_come =216.65k,按照下面计算出来得Re是42.13191825765579
        T_come = 216.65
        C = 110.4/T_come
        eta_come = T_come**(2/3)*(1+C)/(T_come-C)
        Re = 0.0889099*397.953*0.001*300/eta_come
        Ma = 5
        T = gamma * (Ma ** 2) * P / Rho
        eta = (T ** (2 / 3)) * (1 + C) / (T + C)
        lambda_1 = -2 * eta / 3


        rhoU_x = tf.gradients(Rho * U, X)[0]
        rhoV_y = tf.gradients(Rho * V, Y)[0]
        rhoW_z = tf.gradients(Rho * W, Z)[0]
        f_1 = rhoU_x + rhoV_y + rhoW_z

        U_x = tf.gradients(U, X)[0]
        U_y = tf.gradients(U, Y)[0]
        U_z = tf.gradients(U, Z)[0]
        V_x = tf.gradients(V, X)[0]
        V_y = tf.gradients(V, Y)[0]
        V_z = tf.gradients(V, Z)[0]
        W_x = tf.gradients(W, X)[0]
        W_y = tf.gradients(W, Y)[0]
        W_z = tf.gradients(W, Z)[0]

        tao_xx = lambda_1 * (U_x + V_y + W_z) + 2 * eta * U_x
        tao_yy = lambda_1 * (U_x + V_y + W_z) + 2 * eta * V_y
        tao_zz = lambda_1 * (U_x + V_y + W_z) + 2 * eta * W_z

        tao_xy = eta * (V_x + U_y)
        tao_xz = eta * (W_x + U_z)
        tao_yz = eta * (W_y + V_z)


        kappa = eta / ((gamma - 1) * Ma * Ma* Pr)

        T_x = tf.gradients(T, X)[0]
        T_y = tf.gradients(T, Y)[0]
        T_z = tf.gradients(T, Z)[0]
        qx = kappa * T_x
        qy = kappa * T_y
        qz = kappa * T_z

        e = (P / (gamma - 1)) + (0.5 * Rho * (U ** 2 + V ** 2 + W ** 2))
        f2_l1 = tf.gradients(Rho * (U ** 2) + P, X)[0]
        f2_l2 = tf.gradients(Rho * U * V, Y)[0]
        f2_l3 = tf.gradients(Rho * U * W, Z)[0]
        f2_r = tf.gradients(tao_xx, X)[0] + tf.gradients(tao_xy, Y)[0] + tf.gradients(tao_xz, Z)[0]
        f_2 =  Re * (f2_l1 + f2_l2 + f2_l3) - f2_r

        f3_l1 = tf.gradients(Rho * U * V, X)[0]
        f3_l2 = tf.gradients(Rho * (V ** 2) + P, Y)[0]
        f3_l3 = tf.gradients(Rho * V * W, Z)[0]
        f3_r = tf.gradients(tao_xy, X)[0] + tf.gradients(tao_yy, Y)[0] + tf.gradients(tao_yz, Z)[0]
        f_3 = Re * (f3_l1 + f3_l2 + f3_l3) - f3_r

        f4_l1 = tf.gradients(Rho * U * W, X)[0]
        f4_l2 = tf.gradients(Rho * V * W, Y)[0]
        f4_l3 = tf.gradients(Rho * (W ** 2) + P, Z)[0]
        f4_r = tf.gradients(tao_xy, X)[0] + tf.gradients(tao_yz, Y)[0] + tf.gradients(tao_zz, Z)[0]
        f_4 = Re * (f4_l1 + f4_l2 + f4_l3) - f4_r

        f5_l1 = tf.gradients((e + P) * U, X)[0]
        f5_l2 = tf.gradients((e + P) * V, Y)[0]
        f5_l3 = tf.gradients((e + P) * W, Z)[0]
        f5_r1 = tf.gradients((tao_xx * U + tao_xy * V + tao_xz * W) + qx, X)[0]
        f5_r2 = tf.gradients((tao_xy * U + tao_yy * V + tao_yz * W) + qy, Y)[0]
        f5_r3 = tf.gradients((tao_xz * U + tao_yz * V + tao_zz * W) + qz, Z)[0]
        f_5 = Re * (f5_l1 + f5_l2 + f5_l3) - (f5_r1 + f5_r2 + f5_r3)
        if onWall == True:
            Q = (qx**2 + qy**2+qz**2)**0.5
        else:
            Q = 0
        return U,V,W,P,Rho,Q,f_1, f_2, f_3, f_4,f_5

    def callback(self, loss):
        print('Loss: %.3e ' % loss)


    def train(self, nIter):
        tf_dict = {self.X_tf: self.X,self.Y_tf: self.Y, self.Z_tf: self.Z,
                   self.NQ_tf: self.NQ,
                   self.U_tf: self.U, self.V_tf: self.V,self.W_tf: self.W,
                   self.P_tf: self.P, self.Rho_tf: self.Rho,
                   self.onWall_tf: self.onWall
                   }

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            if it % 10 == 0:

                elapsed = time.time() - start_time

                lU = self.sess.run(self.lu, tf_dict)
                lV = self.sess.run(self.lv, tf_dict)
                lW = self.sess.run(self.lw, tf_dict)
                lP = self.sess.run(self.lp, tf_dict)
                lRho = self.sess.run(self.lrho, tf_dict)
                lQ = self.sess.run(self.lq, tf_dict)
                f_1 = self.sess.run(self.l1, tf_dict)
                f_2 = self.sess.run(self.l2, tf_dict)
                f_3 = self.sess.run(self.l3, tf_dict)
                f_4 = self.sess.run(self.l4, tf_dict)
                f_5 = self.sess.run(self.l5, tf_dict)

                loss_value = self.sess.run(self.loss, tf_dict)


                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                print('U: %.3e, V: %.3e , W: %.3e, P: %.3e,\n'
                      ' Rho: %.3e, q: %.3e\n'
                      'f_1: %.3e, f_2: %.3e, f_3: %.3e, f_4: %.3e, f_5: %.3e'%
                      (lU, lV, lW, lP, lRho,lQ ,f_1, f_2, f_3, f_4,f_5 ))

                # print('U: %.3e, V: %.3e , W: %.3e, P: %.3e,'
                #       ' Rho: %.3e, ' %
                #       (lU, lV, lW, lP, lRho))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
    def predict(self, X,Y,Z):

        tf_dict = {self.X_tf: self.X, self.Y_tf: self.Y, self.Z_tf: self.Z,
                   self.U_tf: self.U, self.V_tf: self.V, self.W_tf: self.W,
                   self.P_tf: self.P, self.Rho_tf: self.Rho
                   }

        U_star = self.sess.run(self.U_pred, tf_dict)
        V_star = self.sess.run(self.V_pred, tf_dict)
        W_star = self.sess.run(self.W_pred, tf_dict)
        P_star = self.sess.run(self.P_pred, tf_dict)
        Rho_star = self.sess.run(self.Rho_pred, tf_dict)
        NQ_star = self.sess.run(self.NQ_pred, tf_dict)
        return U_star,V_star,W_star,P_star,Rho_star,NQ_star
        return U_star,V_star,W_star,P_star,Rho_star

def loadData(file):
    data = np.load(file)
    return data.item()

def npformat(X):
    Y = np.array(X).flatten()[:, None]
    return Y
def getBaseVar(idx , list):
    return list[idx,:]
def Dimensionless(standerd,var):
    return var/standerd

if __name__ == "__main__":
    datafile = "F:\PINNs_rewrite\A__bluntcone_a-05_H20km_Ma05_0.00086mm\A00.npy"

    data = loadData(datafile)
    hotData = data["hotData"]
    tecData = data["tecData"]
    # 壁面上的数据
    NDimQ = hotData["NDim_Q"]
    Nw_data = len(NDimQ)  # 壁面上的数据量
    # 流场中的数据
    X = tecData["X"]
    Y = tecData["Y"]
    Z = tecData["Z"]
    U = tecData["U"]
    V = tecData["V"]
    W = tecData["W"]
    P = tecData["P"]
    Rho = tecData["Rho"]
    Nf_data = len(U)

    # 整理场数据整合
    # 表示是否是壁面上点
    onWall = [True] * Nw_data
    notOnWall = [False] * (Nf_data-Nw_data)
    onWall.extend(notOnWall)
    #整理齐Q的格式
    NDimQ.extend([0]*(Nf_data-Nw_data))

    layers = [3, 20, 20, 20, 20, 20, 20, 3]
    X,Y,Z,NDimQ,U,V,W,P,Rho,onWall= map(npformat,[X,Y,Z,NDimQ,U,V,W,P,Rho,onWall])
# 无量纲化,使所有数据绝对值范围在0-1附近
# 对XYZ，数据在正负300左右，设定L为300
    L = 300
    X, Y, Z = map(Dimensionless,[L]*3,[X, Y, Z])
#  Q已经是无量纲，范围在0 - 0.0045x...
#  估计原UVW的单位是km/s,数据范围在0.X-1.X
    V_come = 397.953*0.001
    U, V, W  =map(Dimensionless,[V_come]*3,[U, V, W])
#  P得范围在0.028x-0.994之间，已经满足数据绝对值范围在0-1附近得要求
#  Rho 0.49x-23.x范围，无量纲化后范围距离0-1更远，暂时不变
    Rho_come = 0.0889099
# a, H, Ma, r, A, onWall条件不做处理
#     获得训练数据,训练数据选取4/5
    N_train = int (Nf_data * 0.8)
    idx = np.random.choice(Nf_data, N_train, replace=False)

    X_train,Y_train,Z_train,\
    NQ_train,U_train,V_train,W_train,\
    P_train,Rho_train,\
    onWall_train\
        =map(getBaseVar,[idx]*10,[X,Y,Z,NDimQ,U,V,W,P,Rho,onWall])

    model = PhysicsInformedNN(X_train,Y_train,Z_train,\
    NQ_train,U_train,V_train,\
    W_train,P_train,Rho_train,\
    onWall_train,\
    layers)

    model.train(100000)

    testSet = set(range(Nf_data)) - set(idx)

    X_test, Y_test, Z_test, Q_test, \
    U_test, V_test, W_test, P_test, Rho_test, \
        = \
        map(getBaseVar, list(testSet) * 14, [X, Y, Z, U, V, W, P, Rho])

    U_pred, V_pred, W_pred, P_pred, Rho_pred, Q_pred = \
        model.predict(X_test, Y_test, Z_test)

    # Error
    error_U = np.linalg.norm(U_test - U_pred, 2) / np.linalg.norm(U_test, 2)
    error_V = np.linalg.norm(V_test - V_pred, 2) / np.linalg.norm(V_test, 2)
    error_W = np.linalg.norm(W_test - W_pred, 2) / np.linalg.norm(W_test, 2)
    error_P = np.linalg.norm(P_test - P_pred, 2) / np.linalg.norm(P_test, 2)
    error_Rho = np.linalg.norm(Rho_test - Rho_pred, 2) / np.linalg.norm(Rho_test, 2)
    error_Q = np.linalg.norm(Q_test - Q_pred, 2) / np.linalg.norm(Q_test, 2)

    print('Error u: %e' % (error_U))
    print('Error v: %e' % (error_V))
    print('Error W: %e' % (error_W))
    print('Error P: %e' % (error_P))
    print('Error Rho: %e' % (error_Rho))
    print('Error Q: %e' % (error_Q))


