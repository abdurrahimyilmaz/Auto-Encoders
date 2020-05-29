# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:03:32 2020

@author: Abdurrahim
"""
#burada stacked auto encoder uygulaması yapılacak
#basit ve komplex şeyler yapabiliyor

#importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing dataset
#sep, dosyalarda veriler nasıl sınıflandırılmış
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#preparing the training and test set
#K katlamalı çarpraz doğrulama için k-kere yöntem çalıştırılır. Her adımda veri kümesinin 1/k kadar, 
#daha önce test için kullanılmamış parçası, test için kullanılırken, geri kalan kısmı eğitim için 
#kullanılır.Literatürde genelde k 10 seçilir.burada onun için 5 tane farklı u datası var
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#getiing the number of users and movies
#5 parçaya ayırdığımız için hangi uda max sayı kaç onu elde ediyoruz
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#converting the data into an array with users in lines and movies in columns
#kullanıcılar ile filmlere verdikleri ratingler eşleştiriliyor karmaşık halden düzenli hale getiriliyor
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1): #1den başla sonuna kadar devam et
        id_movies = data[:,1][data[:,0] == id_users] #data içerisinde 0. column 1. user eşit olduğunda onu 1. columna eşitle
        id_ratings = data[:,2][data[:,0] == id_users]#data içerisinde 0. column 1. user eşit olduğunda onu 2. columna eşitle
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings 
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set) #normalde x user y filmine şu puanı vermiş şeklinde bir taplomuz var
test_set = convert(test_set) #bu fonksiyonla 943 kişinin 1682 film için rating eşleştirilmesi liste içinde toplanmış oldu

#converting the data into torch tensors - torch tensorları numpy göre daha etkili daha basit
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#creating the architecture of the neural network
class SAE(nn.Module): #pytorchdan inherit ile classı kuruyoruz - pythonda sınıflar için capital letter kullanılır 
    def __init__(self, ): #virgülden sonra değişkenleri yazmamıza gerek yok kendisi inherit ediyor
        super(SAE, self).__init__() #optimize için super function kullanılır
        self.fc1 = nn.Linear(nb_movies, 20) #full connection olarak input layerı başlatıyoruz - input number of features yani nb_movies
        self.fc2 = nn.Linear(20, 10) #bu 20 10 vs yaklaşık tune edilebilir experimental değerler
        self.fc3 = nn.Linear(10, 20) #decoding başlattık
        self.fc4 = nn.Linear(20, nb_movies) #çıkışta girişin aynısı kadar node olacak
        self.activation = nn.Sigmoid() #activation olarak sigmoid kullandık
    def forward(self, x): #girişle çıkıştaki predicti kıyasladığımız fonksiyon her katmanda aktivasyon yapılır encoding decoding için
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x #tahmin edilen ratingleri döndürür
    
sae = SAE()
criterion = nn.MSELoss() #loss function için mean square error yaklaşımı kullanılır
#optimizer experimental seçilir - lr = learning rate - weight_decay = her epch sonrası lr düşürmek için kullanılır yakınsamayı yoluna koyar
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#training the sae
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 #başta lossumuz 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) #zero - index of dimenson -- tek ekseni input olarak kabul edimez onun için batch için fake 2. dimension ekliyoruz
        target = input.clone() #başta ikisi aynı
        if torch.sum(target.data > 0) > 0: #memory optimize etmek için bunu yapıyoruz en az bir filmi oylayan kullanıcıları dahil ediyoruz
            output = sae(input)
            target.require_grad = False #targetla ilişkili gradientleri hesaplamıyoruz - kod optimizasyonu için
            output[target == 0] = 0 #hesap hatalarını düşürmek için bunu yapıyoruz böylece bu 0 değerleri katılmayarak ağırlık ayarlaması daha düzgün yapılıyor
            loss = criterion(output, target) # loss function
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #biz sadece epochta ratingli değerleri alıyoruz ama hesap yapılırken ratingsiz değerlerde
            #hesaba katıldığı için ortalamanın düzeltilmesi gerekiyor - 1e-10da payda da oldupu için 0 durumunda hata almamamız için
            loss.backward() #ağırlıklar artacak mı azalacak mı onu belirler
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step() #ağırlıkları ayarlarız - backward yönü bu miktarı belirler
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
    #epoch lossumuz 1in altına inip ama 1e yakın olmalı eğer 1i geçerse çıkışta direk girişi
    #görüyoruz demektir bu da aede cheat yapıyoruz olur 1e yakın olması da düzgün tahmin
    #ettiğimizi gösterir
        
#testing the sae
test_loss = 0 #başta lossumuz 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) #zero - index of dimenson -- tek ekseni input olarak kabul edimez onun için batch için fake 2. dimension ekliyoruz
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0: #memory optimize etmek için bunu yapıyoruz en az bir filmi oylayan kullanıcıları dahil ediyoruz
        output = sae(input)
        target.require_grad = False #targetla ilişkili gradientleri hesaplamıyoruz - kod optimizasyonu için
        output[target == 0] = 0 #hesap hatalarını düşürmek için bunu yapıyoruz böylece bu 0 değerleri katılmayarak ağırlık ayarlaması daha düzgün yapılıyor
        loss = criterion(output, target) # loss function
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #biz sadece epochta ratingli değerleri alıyoruz ama hesap yapılırken ratingsiz değerlerde
        #hesaba katıldığı için ortalamanın düzeltilmesi gerekiyor - 1e-10da payda da oldupu için 0 durumunda hata almamamız için
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.         
print('test loss: ' + str(test_loss/s)) #0.9521 oranında doğruluğumuz var