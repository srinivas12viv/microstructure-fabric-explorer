from numpy import linalg as LA
import subprocess
import numpy as np
import multiprocessing as mp
import psutil

N_cores=input("No. of Cores to use for minskowski computation: ")
print "w020 Reference point for computation?"
print "If reference_origin- Press 0"
print "If reference_Centroid- Press 1"
ref=input("0 or 1 : ")	

while ref!=0 and ref!=1:
  print "Error: Entered other than 0 and 1, please enter correct value"
  ref=input("0 or 1 : ") 
	
f = open ("Vertices.txt", "r")
l = []
l = [ line.split() for line in f]
l = tuple(l)
f.close()

g = open ("Faces.txt", "r")
m = []	
m = [ line.split() for line in g]
m = tuple(m)
g.close()

N_voro=len(l)


f = open ("w020_tensor.txt" , "w")
f.write("ID w020_xx w020_xy w020_xz w020_yy w020_yz w020_zz\n")
f.close()

f = open ("eigen_w020.txt" , "w")
f.write("ID eig_val1 vec1x vec1y vec1z eig_val2 vec2x vec2y vec2z eig_val3 vec3x vec3y vec3z\n")
f.close()



def MT(l,m,d,ref,affinity):

	  t=mp.current_process().name
	  proc=psutil.Process()
	  proc.cpu_affinity(affinity)
          ID=affinity[0]*d
	  counter=0
	  for i in range (0,1): #len(l)
	    s_n1=[]
	    s_n2=[]
	    for j in range(0,len(l[i])):
	
	      s=list(l[i][j])
	      s=s[1:-1]
	      s="".join(s)
	      s_n=s.split(",")
	      s_n1.append(s_n)
	  
	    for j in range(0,len(m[i])):
	
	      s=list(m[i][j])
	      s=s[1:-1]
	      s="".join(s)
	      s_n=s.split(",")
	      s_n2.append(s_n)
	
	    s_n1=np.asarray(s_n1)
	    s_n1=s_n1.astype(np.float)
	    points=tuple(s_n1)
	    faces=tuple(s_n2)

	    #Define Possible triangulation
	    triangles=[]
	    for x in faces:
	      #print points[int(x[0])]
	      for y in range(1,len(x)-1):
	        triangles.append((points[int(x[0])],points[int(x[y])],points[int(x[y+1])]))
  	 
	    normal=[]
	    area=[]
	    com=[]
	    w010=np.zeros(3)
	    w000=0
	    print triangles[0]
	    for x in triangles:
	      vertex_i = x[0]
	      vertex_i1= x[1]
	      vertex_i2= x[2]
	      v=(vertex_i1-vertex_i)
	      w=(vertex_i2-vertex_i)
	      com_temp=(vertex_i+vertex_i1+vertex_i2)/3.
	      normal_temp=np.cross(v,w)/LA.norm(np.cross(v,w))
	      area_temp=LA.norm(np.cross(v,w))/2
	
	      w000+=(np.dot(com_temp,normal_temp)*area_temp*(1/3.))
	
	      for k in range(0,3):
	        part1 = 2*v[k]*v[(k+1)%3] + v[(k+1)%3]*w[k] + 4*vertex_i[(k+1)%3]*(v[k]+w[k]) + v[k]*w[(k+1)%3] + 2*w[k]*w[(k+1)%3] + 4*vertex_i[k]*(3*vertex_i[(k+1)%3]+v[(k+1)%3]+w[(k+1)%3])
	        vf = np.cross(v,w)[(k+1)%3]
	        w010[k]+=vf*part1/ 24.
    
	      area.append(area_temp)
	      normal.append(normal_temp)
	      com.append(com_temp)
	  
	    ref_vector=w010/w000
	    if ref==0:
              ref_vector=[0,0,0]
	    w020=np.zeros((3,3))
    
	    for x in triangles:
	      c1 = (x[0]-ref_vector)
	      c2= (x[1]-ref_vector)
	      c3= (x[2]-ref_vector)
	      v=(c2-c1)
	      w=(c3-c1)
	      n=np.cross(v,w)/LA.norm(np.cross(v,w))
	      a=LA.norm(np.cross(v,w))/2
	      Ixx = (3*pow(c2[0],2)*c2[2] + 2*c2[0]*c2[2]*c3[0] + c2[2]*pow(c3[0],2) + c1[2]*(pow(c2[0],2) + c2[0]*c3[0] + pow(c3[0],2)) + pow(c2[0],2)*c3[2] + 2*c2[0]*c3[0]*c3[2] + 3*pow(c3[0],2)*c3[2] + pow(c1[0],2)*(3*c1[2] + c2[2] + c3[2]) + c1[0]*(2*c1[2]*(c2[0] + c3[0]) + c2[0]*(2*c2[2] + c3[2]) + c3[0]*(c2[2] + 2*c3[2])))/60.
	      w020[0,0] += Ixx*2*a*n[2]

      	      Iyy = (3*pow(c2[1],2)*c2[2] + 2*c2[1]*c2[2]*c3[1] + c2[2]*pow(c3[1],2) + c1[2]*(pow(c2[1],2) + c2[1]*c3[1] + pow(c3[1],2)) + pow(c2[1],2)*c3[2] + 2*c2[1]*c3[1]*c3[2] + 3*pow(c3[1],2)*c3[2] + pow(c1[1],2)*(3*c1[2] + c2[2] + c3[2]) + c1[1]*(2*c1[2]*(c2[1] + c3[1]) + c2[1]*(2*c2[2] + c3[2]) + c3[1]*(c2[2] + 2*c3[2])))/60.
	      w020[1,1] += Iyy*2*a*n[2]

	      Izz = (3*c2[1]*pow(c2[2],2) + pow(c2[2],2)*c3[1] + pow(c1[2],2)*(c2[1] + c3[1]) + 2*c2[1]*c2[2]*c3[2] + 2*c2[2]*c3[1]*c3[2] + c2[1]*pow(c3[2],2) + 3*c3[1]*pow(c3[2],2) + c1[1]*(3*pow(c1[2],2) + pow(c2[2],2) + c2[2]*c3[2] + pow(c3[2],2) + 2*c1[2]*(c2[2] + c3[2])) + c1[2]*(c2[1]*(2*c2[2] + c3[2]) + c3[1]*(c2[2] + 2*c3[2])))/60.
	      w020[2,2] += Izz*2*a*n[1]

	      Ixy = (2*c1[2]*c2[0]*c2[1] + 6*c2[0]*c2[1]*c2[2] + c1[2]*c2[1]*c3[0] + 2*c2[1]*c2[2]*c3[0] + c1[2]*c2[0]*c3[1] + 2*c2[0]*c2[2]*c3[1] + 2*c1[2]*c3[0]*c3[1] + 2*c2[2]*c3[0]*c3[1] + 2*c2[0]*c2[1]*c3[2] + 2*c2[1]*c3[0]*c3[2] + 2*c2[0]*c3[1]*c3[2] + 6*c3[0]*c3[1]*c3[2] + c1[0]*(2*c2[1]*c2[2] + c2[2]*c3[1] + 2*c1[2]*(c2[1] + c3[1]) + c2[1]*c3[2] + 2*c3[1]*c3[2] + 2*c1[1]*(3*c1[2] + c2[2] + c3[2])) + c1[1]*(2*c1[2]*(c2[0] + c3[0]) + c2[0]*(2*c2[2] + c3[2]) + c3[0]*(c2[2] + 2*c3[2])))/120.

	      w020[0,1] += Ixy*2*a*n[2]
	      w020[0,2] += Ixy*2*a*n[1]
	      w020[1,2] += Ixy*2*a*n[0]

	    w020[1,0]=w020[0,1]
	    w020[2,0]=w020[0,2]
	    w020[2,1]=w020[1,2]
	 
	    if np.isnan(w020[0,0])==True:
	      counter=counter+1
	    else:  
	      ed_val,ed_vec=LA.eig(w020)
	      ed_vec=np.transpose(ed_vec)

	      f=open("w020_tensor.txt" ,"a")
	      print >> f, "%s %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f" % (ID+i, w020[0,0],w020[0,1],w020[0,2],w020[1,1],w020[1,2],w020[2,2])
	      f.close() 
	 
	      f=open("eigen_w020.txt" ,"a")
	      print >> f, "%s %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f %0.9f" % (ID+i, ed_val[0],ed_vec[0][0],ed_vec[0][1],ed_vec[0][2],ed_val[1],ed_vec[1][0],ed_vec[1][1],ed_vec[1][2],ed_val[2],ed_vec[2][0],ed_vec[2][1],ed_vec[2][2])
	      f.close()   
	    
	  
#------------------------Parallelization on N-Cores-------------------------------------------------------------------#
p=[]

Dist=int(N_voro/N_cores)             #Distributing different Voronoi cells to different cores

for i in range(0,N_cores):
  if i!=N_cores-1:
    p.append(mp.Process(name=str(i+1),target=MT,args=(l[Dist*i:Dist*(i+1)],m[Dist*i:Dist*(i+1)],Dist,ref,[i])))     
  if i==N_cores-1:
    p.append(mp.Process(name=str(i+1),target=MT,args=(l[Dist*i:],m[Dist*i:],Dist,ref,[i])))
    	  
for i in range(0,N_cores):
  p[i].start()
	
for i in range(0,N_cores):
  p[i].join()
