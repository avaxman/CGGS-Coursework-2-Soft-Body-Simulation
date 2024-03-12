import cppyy

isCollide_code="""

#include "ccd.h"

void support(const void *_obj, const ccd_vec3_t *_d, ccd_vec3_t *_p);
void stub_dir(const void *obj1, const void *obj2, ccd_vec3_t *dir);
void center(const void *_obj, ccd_vec3_t *dir);

struct SmallMesh{
    int numVertices;
    const double* vertices;
    const double* COM;
};

bool isCollide(const double* n1, const double* v1, const double* COM1, const double* n2, const double* v2,  const double* COM2, double*  depth, double* intNormal, double* intPosition){
       
    ccd_t ccd;
    CCD_INIT(&ccd);
    ccd.support1       = support; // support function for first object
    ccd.support2       = support; // support function for second object
    ccd.center1         =center;
    ccd.center2         =center;
    
    ccd.first_dir       = stub_dir;
    ccd.max_iterations = 100;     // maximal number of iterations
    
    SmallMesh m1,m2;
    m1.numVertices = int(*n1);
    m1.vertices = v1;
    m1.COM = COM1;
    m2.numVertices = int(*n2);
    m2.vertices = v2;
    m2.COM = COM2;
    
    void* obj1=(void*)&m1;
    void* obj2=(void*)&m2;
    
    ccd_real_t _depth;
    ccd_vec3_t dir, pos;
    
    int nonintersect = ccdMPRPenetration(obj1, obj2, &ccd, &_depth, &dir, &pos);
    
    if (nonintersect)
      return false;
    
    for (int k=0;k<3;k++){
      intNormal[k]=dir.v[k];
      intPosition[k]=pos.v[k];
    }
    
    depth[0] =_depth;
    for (int k=0;k<3;k++)
        intPosition[k]-=depth[0]*intNormal[k]/2.0;
    
    //Vector3d p1=intPosition+depth*intNormal;
    //Vector3d p2=intPosition;
    //std::cout<<"intPosition: "<<intPosition<<std::endl;
    
    //std::cout<<"depth: "<<depth<<std::endl;
    //std::cout<<"After ccdGJKIntersect"<<std::endl;
    
    //return !nonintersect;
    
    return true;
    
  }
  
/*****************************Auxiliary functions for collision detection. Do not need updating********************************/

/** Support function for libccd*/
void support(const void *_obj, const ccd_vec3_t *_d, ccd_vec3_t *_p)
{
  // assume that obj_t is user-defined structure that holds info about
  // object (in this case box: x, y, z, pos, quat - dimensions of box,
  // position and rotation)
  //std::cout<<"calling support"<<std::endl;
  SmallMesh *obj = (SmallMesh *)_obj;
  //RowVector3d p;
  double d[3];
  for (int i=0;i<3;i++)
    d[i]=_d->v[i]; //p(i)=_p->v[i];
  
  double norm = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
  for (int i=0;i<3;i++)
    d[i]=d[i]/norm;
  //std::cout<<"d: "<<d<<std::endl;
  
  int maxVertex=-1;
  int maxDotProd=-32767.0;
  for (int i=0;i<obj->numVertices;i++){
    double currVec[3];
    for (int j=0;j<3;j++)
        currVec[j] = obj->vertices[i*3+j]-obj->COM[j];
    
    double currDotProd = 0.0;    
    for (int j=0;j<3;j++)
        currDotProd +=currVec[j]*d[j];
    
    //double currDotProd=d.dot(obj->vertices.row(i)-RowVector3d(obj->COM[0],obj->COM[1],obj->COM[2]));
    if (maxDotProd < currDotProd){
      maxDotProd=currDotProd;
      //std::cout<<"maxDotProd: "<<maxDotProd<<std::endl;
      maxVertex=i;
    }
    
  }
  //std::cout<<"maxVertex: "<<maxVertex<<std::endl;
  
  for (int i=0;i<3;i++)
    _p->v[i]=obj->vertices[3*maxVertex+i];
  
  //std::cout<<"end support"<<std::endl;
}

void stub_dir(const void *obj1, const void *obj2, ccd_vec3_t *dir)
{
  dir->v[0]=1.0;
  dir->v[1]=0.0;
  dir->v[2]=0.0;
}

void center(const void *_obj,ccd_vec3_t *center)
{
  SmallMesh *obj = (SmallMesh *)_obj;
  for (int i=0;i<3;i++)
    center->v[i]=obj->COM[i];
}

"""
cppyy.cppdef(isCollide_code)


