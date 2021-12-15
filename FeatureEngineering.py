
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import time
import csv
import random
import os
from numpy import array
import math
import pandas as pd
from BKT import BKT

data_name= '4_Ass_09'
batch_size=32

def k_means_clust(session, train_students, test_students, max_stu, max_seg, num_clust, num_skills, num_iter):
    identifiers=3
    max_stu=int(max_stu)
    max_seg=int(max_seg)
    cluster= np.zeros((max_stu,max_seg))
    data=[]
    for ind,i in enumerate(train_students):
        data.append(i[:-identifiers])
    data = array(data)
    points = tf.constant(data)  

    centroids = tf.Variable(tf.random_shuffle(points)[:num_clust, :])
    # calculate distances from the centroids to each point
    points_e = tf.expand_dims(points, axis=0) # (1, N, 2)
    centroids_e = tf.expand_dims(centroids, axis=1) # (k, 1, 2)  
    distances = tf.reduce_sum((points_e - centroids_e) ** 2, axis=-1) # (k, N)
    # find the index to the nearest centroids from each point
    indices = tf.argmin(distances, axis=0) # (N,)
    # gather k clusters: list of tensors of shape (N_i, 1, 2) for each i
    clusters = [tf.gather(points, tf.where(tf.equal(indices, i))) for i in range(num_clust)]
    # get new centroids (k, 2)
    new_centroids = tf.concat([tf.reduce_mean(clusters[i], reduction_indices=[0]) for i in range(num_clust)], axis=0)
    # update centroids
    assign = tf.assign(centroids, new_centroids)
    session.run(tf.global_variables_initializer())
    for j in range(num_iter):
        clusters_val, centroids_val, _ = session.run([clusters, centroids, assign])
        
    
    for ind,i in enumerate(train_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None            
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j])< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j])
               if cur_dist<min_dist:                  
                  min_dist=cur_dist
                  closest_clust=j                  
        
        cluster[int(i[-2]),int(i[-1])]=closest_clust
        
   
    for ind,i in enumerate(test_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None 
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j])< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j])
               if cur_dist<min_dist:
                  min_dist=cur_dist
                  closest_clust=j
        cluster[int(i[-2]),int(i[-1])]=closest_clust
        
        
    del train_students, test_students
    
    return cluster
  
    
def difficulty_data(students,max_items):
          
    limit= 3
    xtotal = np.zeros(max_items+1)
    x1 = np.zeros(max_items+1)
    items=[]
    Allitems=[]
    item_diff ={} 
    index=0      
    while(index < len(students)):
         student = students[index]         
         item_ids = student[3]
         correctness = student[2]         
         for j in range(len(item_ids)):         
             
             key =item_ids[j]             
             xtotal[key] +=1
             if(int(correctness[j]) == 0):
                x1[key] +=1
             if xtotal[key]>limit and key > 0 and key not in items  :
                items.append(key)
             
             if xtotal[key]>0 and key not in Allitems :
                Allitems.append(key)
                
         index+=1
    for i in (items):
        diff =(np.around(float(x1[i])/float(xtotal[i]), decimals=1)*10).astype(int)   
        item_diff[i]=diff
    

    return item_diff     

def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
    




def read_data_from_csv_file(trainfile, testfile):
    rows = []
    max_skills = 0
    max_steps = 0 
    max_items =0
    studentids = []
    train_ids=[]
    test_ids=[]
    
    problem_len = 20  
    with open(trainfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
            
            
    skill_rows=[]
    correct_rows=[]
    stu_rows=[]
    opp_rows=[]
    index = 0
    while(index < len(rows)):
         if int(rows[index][0])>problem_len: 
            problems = int(rows[index][0]) 
            student_id= int(rows[index][1])
            train_ids.append(student_id)  
            
            tmp_max_skills = max(map(int, rows[index+1]))
            if(tmp_max_skills > max_skills):
               max_skills = tmp_max_skills
                        
                        
            tmp_max_items = max(map(int, rows[index+2]))
            if(tmp_max_items > max_items):
               max_items = tmp_max_items
               
            skill_rows=np.append(skill_rows,rows[index+1])
            correct_rows=np.append(correct_rows,rows[index+3])
            stu_rows=np.append(stu_rows,([student_id]* len(rows[index+1])))
            opp_rows=np.append(opp_rows, list(range(len(rows[index+1]))))
         index += 4  
         
         
         
    with open(testfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
            
            
    
    while(index < len(rows)):
         if int(rows[index][0])>problem_len: 
            problems = int(rows[index][0]) 
            student_id= int(rows[index][1])
            test_ids.append(student_id)  
            
            tmp_max_skills = max(map(int, rows[index+1]))
            if(tmp_max_skills > max_skills):
               max_skills = tmp_max_skills
                        
                        
            tmp_max_items = max(map(int, rows[index+2]))
            if(tmp_max_items > max_items):
               max_items = tmp_max_items
               
            skill_rows=np.append(skill_rows,rows[index+1])
            correct_rows=np.append(correct_rows,rows[index+3])
            stu_rows=np.append(stu_rows,([student_id]* len(rows[index+1])))
            opp_rows=np.append(opp_rows, list(range(len(rows[index+1]))))
         index += 4  
         
         
         
         
         
         
    max_skills =max_skills+1
    max_items = max_items+1     
    
    data= pd.DataFrame({'stus': stu_rows, 'skills': skill_rows, 'corrects': correct_rows, 'opp': opp_rows}).astype(int)
    bkt_ass= BKTAssessment(data, train_ids, max_skills)
    
    del skill_rows, correct_rows, stu_rows, opp_rows, data
    
    
    index = 0   
    tuple_rows = []
    while(index < len(rows)):
          if int(rows[index][0])>problem_len: 
                  
                  problems = int(rows[index][0]) 
                  student_id= int(rows[index][1])
                  studentids.append(student_id)  
                  
                                 
                  
                  
                  if (problems>problem_len):
                  
                  
                     tmp_max_steps = int(rows[index][0])
                     if(tmp_max_steps > max_steps):
                        max_steps = tmp_max_steps
                        
                        
                     asses= bkt_ass[student_id]
                     
                     
                                        
                     len_problems = int(int(problems) / problem_len)*problem_len
                     rest_problems = problems - len_problems             
                     
                     ele_p = []             
                     p_index=0       
                     for element in rows[index+1]:
                         ele_p.append(int(element))
                         p_index=p_index+1 
                         
                     ele_c = []
                     c_index=0
                     for element in rows[index+3]:
                         ele_c.append(int(element))
                         c_index=c_index+1
                         
                         
                     ele_d = []
                     d_index=0
                     for element in rows[index+2]:
                         ele_d.append(int(element))
                         d_index=d_index+1
                         
                         
                     ele_a = []             
                     a_index=0       
                     for element in asses:
                         ele_a.append(float(element))
                         a_index=a_index+1 

                     if (rest_problems>0):
                        rest=problem_len-rest_problems
                        for i in range(rest):
                            ele_p.append(-1)
                            ele_c.append(-1)
                            ele_d.append(-1)
                            ele_a.append(-1)

                     ele_p_array = np.reshape(np.asarray(ele_p), (-1,problem_len))
                     ele_c_array = np.reshape(np.asarray(ele_c), (-1,problem_len))
                     ele_d_array = np.reshape(np.asarray(ele_d), (-1,problem_len))
                     ele_a_array = np.reshape(np.asarray(ele_a), (-1,problem_len))
                   
                     n_pieces = ele_p_array.shape[0]
                     
                   
                     for j in range(n_pieces):
                         s1=[student_id,j,problems]
                         
                         if (j>-1) & (j< (n_pieces-1)) :
                            s1.append(1)
                            s2= np.append(ele_p_array[j,:],ele_p_array[j+1,0]).tolist()
                            s3= np.append(ele_c_array[j,:],ele_c_array[j+1,0]).tolist() 
                            s4= np.append(ele_d_array[j,:],ele_d_array[j+1,0]).tolist()  
                            s5= np.append(ele_a_array[j,:],ele_a_array[j+1,0]).tolist()      
                         else:
                              s1.append(-1)
                              s2= ele_p_array[j,:].tolist()
                              s3= ele_c_array[j,:].tolist() 
                              s4= ele_d_array[j,:].tolist() 
                              s5= ele_a_array[j,:].tolist() 
                         tup = (s1,s2,s3,s4,s5)
                         tuple_rows.append(tup)
          index += 4
          
          
          
    
    
    
    max_steps  =max_steps+1
    
    
    index=0
    train_students=[]
    test_students=[]            
    while(index < len(tuple_rows)):
         if (int(tuple_rows[index][0][0]) in train_ids):
            train_students.append(tuple_rows[index])
         if (int(tuple_rows[index][0][0]) in test_ids):
            test_students.append(tuple_rows[index])
         index+=1
    
      
    return train_students, test_students, studentids, max_skills, max_items, train_ids, test_ids
    
    
def get_bktdata(df): 
    BKT_dict = {}
    DKT_skill_dict = {}
    DKT_res_dict = {}   

    for kc in list(df['skills'].unique()):
        kc_df=df[df['skills']==kc].sort_values(['stus'],ascending=True)             
        stu_cfa_dict = {}
        
        for stu in list(kc_df['stus'].unique()):
            df_final=kc_df[kc_df['stus']==int(stu)].reset_index().sort_values(['opp'],ascending=True)
            stu_cfa_dict[int(stu)]=list(df_final['corrects'])
            
        BKT_dict[int(kc)]=stu_cfa_dict
        
        
    for stu in list(df['stus'].unique()):
        stu_df=df[df['stus']==int(stu)].sort_values(['opp'],ascending=True)
        DKT_skill_dict[int(stu)]=list(stu_df['skills'])
        DKT_res_dict[int(stu)]=list(stu_df['corrects'])
        
    

    return BKT_dict, DKT_skill_dict, DKT_res_dict
            
        

def cluster_data(students,max_stu,num_skills, datatype):

    
    

    success = []
    max_seg =0    
    xtotal = np.zeros((max_stu,num_skills))    
    x1 = np.zeros((max_stu,num_skills))
    x0 = np.zeros((max_stu,num_skills)) 
    
    index = 0  
    while(index+ batch_size < len(students)):    
         for i in range(batch_size):
             student = students[index+i] 
             student_id = int(student[0][0])
             seg_id = int(student[0][1])
             
                 
             if (int(student[0][3])==1):
                tmp_seg = seg_id
                if(tmp_seg > max_seg):
                   max_seg = tmp_seg
                problem_ids = student[1]                
                correctness = student[2]
                for j in range(len(problem_ids)):           
                    key =problem_ids[j]
                    xtotal[student_id,key] +=1
                    if(int(correctness[j]) == 1):
                      x1[student_id,key] +=1
                    else:
                         x0[student_id,key] +=1

                
                xsr=[(x+1.4)/(y+2)  for x, y in zip(x1[student_id], xtotal[student_id])]
                
                x=np.nan_to_num(xsr)
                x=np.append(x, student_id)
                x=np.append(x, seg_id)
                success.append(x) 
                
                
         index += batch_size 
         
    return success, max_seg 
    
    
    
   
def BKTAssessment(data, train_ids, max_skills):

    bkt_data, dkt_skill, dkt_res =get_bktdata(data)
    DL, DT, DG, DS ={}, {}, {}, {}
    for i in bkt_data.keys():
        skill_data = bkt_data[i]
        train_data=[]
        for j in skill_data.keys():
            if int(j) in train_ids:                       
               train_data.append(list(map(int,skill_data[j])))
                       
        bkt = BKT(step = 0.1, bounded = False, best_k0 = True)
        if len(train_data)>2:
           DL[i],DT[i],DG[i],DS[i]=bkt.fit(train_data)   
        else:
             DL[i],DT[i],DG[i],DS[i] = 0.5, 0.2, 0.1, 0.1   
        
    del bkt_data
    
    mastery =  bkt.inter_predict(dkt_skill, dkt_res, DL, DT, DG, DS, max_skills)
    
    del dkt_skill, dkt_res
    
    return mastery
    
    print("**************Finished BKT Assessment****************")
    
    
    
def get_features(students, item_diff, max_stu, cluster, num_skills, datatype):
    """Runs the model on the given data."""   
    index = 0
    
    stu_list=[]
    p0_list=[]
    p1_list=[]
    p2_list=[]
    p3_list=[]
    p4_list=[]
    
        
    while(index+ batch_size < len(students)):
    
        for i in range( batch_size):
            student = students[index+i]
            student_id = student[0][0]
            seg_id = int(student[0][1]) 

            ## assign cluster of student at segment z-1
            ## seg_id==0 is initial segment with initial unidentified cluster for all student
            if (seg_id>0):
                cluster_id= cluster[student_id,(seg_id-1)]+1
            else:
                cluster_id= 0           
            
            skill_ids = student[1]
            correctness = student[2] 
            items = student[3]       
            bkt= student[4] 
            
            
                
           
                        
            for j in range(len(skill_ids)-1):
            
                            
                target_indx = j+1
                skill_id = int(skill_ids[target_indx])
                item = int(items[target_indx]) 
                kcass = np.round(float(bkt[target_indx]),6)
                
                
                correct = int(correctness[target_indx])
                # to ignore if target_id is null or -1 all skill index are started from 0
                
                if skill_id > -1:
                
                   df = 0
                   if item in item_diff.keys():                      
                      df = int(item_diff[item])
                   else:
                        df=5
                   
                   
                   
                   stu_list.append(student_id)
                   p0_list.append(int(skill_id))
                   p1_list.append(float(kcass))
                   p2_list.append(int(cluster_id))
                   p3_list.append(int(df))
                   p4_list.append(int(correct))
                      
                 
        index += batch_size
        
        
        
        
        
        
    
    data= pd.DataFrame({'skill_id': p0_list,'skill_mastery': p1_list, 'ability_profile': p2_list, 'problem_difficulty': p3_list, 'correctness': p4_list})
    data.to_csv ("./"+datatype+"_data.csv", index = None, header=True)
    
    return
    



def main(unused_args):

    
    
    cluster_num= 7
    problem_len= 20
    
    
    train_data='./data/'+data_name+'_train.csv'
    test_data= './data/'+data_name+'_test.csv'
          
    train_students, test_students, student_ids, max_skills, max_items, train_ids, test_ids =read_data_from_csv_file(train_data, test_data)
    num_skills = max_skills
                 
    
    item_diff = difficulty_data(train_students+test_students,max_items)             

    train_cluster_data, train_max_seg= cluster_data(train_students,max(train_ids)+1,max_skills,"train")        
    test_cluster_data, test_max_seg= cluster_data(test_students,max(test_ids)+1,max_skills, "test")
    
    max_stu= max(student_ids)+1
    max_seg=max([int(train_max_seg),int(test_max_seg)])+1
    with tf.Session() as session:

         cluster =k_means_clust(session, train_cluster_data, test_cluster_data, max_stu, max_seg, cluster_num, max_skills, 40)
         get_features(train_students, item_diff, max_stu, cluster, max_skills, "train" )
         get_features(test_students, item_diff, max_stu, cluster, max_skills, "test")
                    
                            

                        

             
               
if __name__ == "__main__":
    tf.app.run()
