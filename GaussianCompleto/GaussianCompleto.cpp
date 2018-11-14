/**********************************************************************************************
*
*       SERGIO GONZALEZ VELAZQUEZ y DAVID CARNEROS PRADO 
*
*   --> Conclusiones:
*			-Aplicando la propiedad de separabilidad, esto es, aplicar el vector vertical a cada
*			punto de la imagen y después aplicar el vector horizontal a la imagen resultante hace que 
*			el programa tenga una ganancia en velocidad de 3,259 (0,541967626s/0,166301223s) respecto
*			al algoritmo naive_matriz que no hace uso de tal propiedad.
*
*			-En lo que se refiere a parelizar mediante OpenMP las funciones que aplican el vector vertical
*			y horizontal, llegamos a las siguientes conclusiones:
*
*				-El programa se ejecuta en 0,125407038 segundos cuando Paralelizamos la función 
*				vect_vertical pero seguimos utilizando la versión secuencial de la funcion que aplica
*				el vector horizontal. Esto significa que se ha conseguido una ganancia en velocidad 
*				respecto a la función separa_Vectores secuencial de 1,32. La ganancia en velocidad respecto 
*				a la funcion naive_matriz  que no aplica la propiedad de separabilidad es de 4,32
*
*				-Sin embargo, al paralelizar la función vect_horizontal y mantener secuencial la función 
*				que aplica el vector vertical no se consigue ninguna mejora significativa respecto a 
*				śepara_vectores secuencial, ya que se obtienen tiempos de ejecución similares.
*
*				-Cuando paralelizamos tanto la función que aplica el vector horizontal como vertical, 
*				el programa se ejecuta en un tiempo similar a cuando paralelizamos únicamente la función
*				que aplica el vector vertical. Luego, podemos concluir que las mejoras en velocidad en 
*				separa_vectores secuencial se deben únicamente a las mejoras conseguidas por paralelizar la 
*				función vect_vertical. 
*
*
*   --> Nota: Los tiempos de ejecucion utilizados para llegar a las conclusiones anteriores
*             han sido obtenidos ejecutando el programa en un computador con 2 nucleos.
*
*             En dicha ejecucion, se ha obtenido la siguiente salida:
*
*				-naive_matriz: 	 										0,541967626 segundos
*               -separa_vectores secuencial:  							0,166301223 segundos
*               -vect_vertical_parallel-vect_horizontal secuencial: 	0,125407038 segundos
*				-vect_vertical secuencial vect_horizontal_parallel: 	0,174392511 segundos
*               -vect_vertical_parallel vect_horizontal_parallel: 		0,125862559 segundos
*
*************************************************************************************************/

#include <QtGui/QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>

#define N 5
#define SEQ 0 
#define PARAL 1
#define SEQ_HOR_PARAL_VERT 2
#define PARAL_HOR_SEQ_VERT 3

const int M=N/2;
int matrizGauss[N][N] =	{{ 1, 4, 6, 4,1 },
						 { 4,16,24,16,4 },
						 { 6,24,36,24,6 },
						 { 4,16,24,16,4 },
						 { 1, 4, 6, 4,1}};
						 
int vectorGauss[N] = { 1, 4, 6, 4, 1 };

int alto, ancho;
#define max(num1, num2) (num1>num2 ? num1 : num2)
#define min(num1, num2) (num1<num2 ? num1 : num2)

double naive_matriz(QImage* image, QImage* result) {
  
  int h, w, i, j;
  int red, green, blue;
  int mini, minj, supi, supj;
  QRgb aux;  
  double start_time = omp_get_wtime();

  for (h = 0; h < alto; h++)
    
    for (w = 0; w < ancho ; w++) {

	  /* Aplicamos el kernel (matrizGauss dividida entre 256) al entorno de cada pixel de coordenadas (w,h)
		 Corresponde multiplicar la componente matrizGauss[i,j] por el pixel de coordenadas (w-M+j, h-M+i)
		 Pero ha de cumplirse 0<=w-M+j<ancho y 0<=h-M+i<alto. Por tanto: M-w<=j<ancho+M-w y M-h<=i<alto+M-h
		 Además, los índices i y j de la matriz deben cumplir 0<=j<N, 0<=i<N
		 Se deduce:  máx(M-w,0) <= j < mín(ancho+M-w,N); máx(M-h,0) <= i < mín(alto+M-h,N)*/
	  red=green=blue=0;
	  mini = max((M-h),0); minj = max((M-w),0);					// Ver comentario anterior
	  supi = min((alto+M-h),N); supj = min((ancho+M-w),N);	    // Íd.
	  for (i=mini; i<supi; i++)
		for (j=minj; j<supj; j++)	{
			aux = image->pixel(w-M+j, h-M+i);
			red += matrizGauss[i][j]*qRed(aux);
			green += matrizGauss[i][j]*qGreen(aux);
			blue += matrizGauss[i][j]*qBlue(aux);
		};

	  red /= 256; green /= 256; blue /= 256;

	  result->setPixel(w,h, QColor(red, green, blue).rgba());
      
	}
  
  return omp_get_wtime() - start_time;    
}	// Fin naive_matriz



void aplicar_vect_vertical(QImage* image,int *red, int *green, int *blue) {
  
  int h, w, i;
  int mini, supi;
  QRgb aux;  
  int indice;
  int redAux,greenAux,blueAux;

  for (h = 0; h < alto; h++){

  	mini = max((M-h),0);
	supi = min((alto+M-h),N);

    for (w = 0; w < ancho ; w++) {
	  
	  redAux=0,greenAux=0,blueAux=0;

	  for (i=mini; i<supi; i++){

			aux = image->pixel(w, h-M+i);

			redAux += vectorGauss[i]*qRed(aux);
			greenAux += vectorGauss[i]*qGreen(aux);
			blueAux += vectorGauss[i]*qBlue(aux);
		};

		indice=(h*ancho + w);

		red[indice] = redAux;
		green[indice] = greenAux;
		blue[indice] = blueAux;
      
	}
  }

 
    
}	

void aplicar_vect_horizontal(int* red, int* green, int* blue, QImage* result) {
  
  int h, w,j;
  int minj, supj; 
  int indice;
  int redAux,greenAux,blueAux;

  for (h = 0; h < alto; h++)
    
    for (w = 0; w < ancho ; w++) {

	  minj = max((M-w),0);			
	  supj = min((ancho+M-w),N);	 
	  	
	  	redAux=0,greenAux=0,blueAux=0;

		for (j=minj; j<supj; j++)	{

			indice = h*ancho + w+j-M;

			redAux += vectorGauss[j]*red[indice];
			greenAux += vectorGauss[j]*green[indice];
			blueAux += vectorGauss[j]*blue[indice];

		};
		 
		 result->setPixel(w,h, QColor(redAux/256, greenAux/256, blueAux/256).rgba());  
	}  
}	

void aplicar_vect_vertical_parallel(QImage* image,int *red, int *green, int *blue) {
  
  int h, w, i;
  int mini, supi;
  QRgb aux;  
  int indice;
  int redAux,greenAux,blueAux;

  #pragma omp parallel for private(w,mini,supi,redAux,greenAux,blueAux,aux,indice,i)

  for (h = 0; h < alto; h++){

  	mini = max((M-h),0);
	supi = min((alto+M-h),N);

  	for (w = 0; w < ancho ; w++) {

	  redAux=0,greenAux=0,blueAux=0;

	  for (i=mini; i<supi; i++){

			aux = image->pixel(w, h-M+i);

			redAux += vectorGauss[i]*qRed(aux);
			greenAux += vectorGauss[i]*qGreen(aux);
			blueAux += vectorGauss[i]*qBlue(aux);
		};

		indice=(h*ancho + w);

		red[indice] = redAux;
		green[indice] = greenAux;
		blue[indice] = blueAux; 
	}
  }
   
}	

void aplicar_vect_horizontal_parallel(int* red, int* green, int* blue, QImage* result) {
  
  int h, w,j;
  int minj, supj; 
  int indice;
  int redAux,greenAux,blueAux;

  #pragma omp parallel for private(w,minj,supj,redAux,greenAux,blueAux,indice,j)
  for (h = 0; h < alto; h++)
    
    for (w = 0; w < ancho ; w++) {

	  minj = max((M-w),0);			
	  supj = min((ancho+M-w),N);	 
	  	
	  	redAux=0,greenAux=0,blueAux=0;

		for (j=minj; j<supj; j++)	{

			indice = h*ancho + w+j-M;

			redAux += vectorGauss[j]*red[indice];
			greenAux += vectorGauss[j]*green[indice];
			blueAux += vectorGauss[j]*blue[indice];

		};
		#pragma omp critical 
		{
		 result->setPixel(w,h, QColor(redAux/256, greenAux/256, blueAux/256).rgba());
		}
	} 
}	


double separa_vectores(QImage* image, QImage* result, int flag){

  double start_time = omp_get_wtime();
   
  int *red,*green,*blue;
  red=(int *)malloc(ancho*alto*sizeof(int));
  green=(int *)malloc(ancho*alto*sizeof(int));
  blue=(int *)malloc(ancho*alto*sizeof(int));

  memset(red,0,sizeof(int)*alto*ancho);
  memset(green,0,sizeof(int)*alto*ancho);
  memset(blue,0,sizeof(int)*alto*ancho);

  switch(flag){
    case SEQ:
    	aplicar_vect_vertical(image,red,green,blue);
    	aplicar_vect_horizontal(red,green,blue,result);
    break;

    case PARAL:
    	aplicar_vect_vertical_parallel(image,red,green,blue);
    	aplicar_vect_horizontal_parallel(red,green,blue,result);
    break;

    case SEQ_HOR_PARAL_VERT:
    	aplicar_vect_vertical_parallel(image,red,green,blue);
    	aplicar_vect_horizontal(red,green,blue,result);
    break;

    case PARAL_HOR_SEQ_VERT:
    	aplicar_vect_vertical(image,red,green,blue);
    	aplicar_vect_horizontal_parallel(red,green,blue,result);
    break;
  }  

  free(red);
  free(green);
  free(blue);

  return omp_get_wtime() - start_time;  
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsView view(&scene);

    if (argc != 2) {printf("Vuelva a ejecutar. Uso: <ejecutable> <archivo imagen> \n"); return -1;} 
    QPixmap qp = QPixmap(argv[1]);
    if(qp.isNull()) { printf("image not found\n"); return -1;}
    
    QImage image = qp.toImage();
    
    alto = image.height(); ancho = image.width();
    
    QImage matrGaussImage(image);
    QImage imageAux(image);

    /*naive_matriz y gauss separa_vectores secuencial*/
    double computeTime = naive_matriz(&image, &matrGaussImage);
    printf("naive_matriz time: %0.9f seconds\n", computeTime);
    
	computeTime = separa_vectores(&image, &imageAux,SEQ);

	printf("separa_vectores secuencial time: %0.9f seconds\n", computeTime);
	if (matrGaussImage == imageAux) printf("Algoritmo gauss naive_matriz y gauss separa_vectores dan la misma imagen\n\n");
	else printf("Algoritmo gauss naive_matriz y gauss separa_vectores dan distinta imagen\n\n");


	/*separa_vectores con vect_vertical_parallel y vect_horizontal secuencial*/
	computeTime = separa_vectores(&image, &imageAux,SEQ_HOR_PARAL_VERT);
	printf("separa_vectores con vect_vertical_parallel y vect_horizontal secuencial time: %0.9f seconds\n", computeTime);

	if (matrGaussImage == imageAux) printf("Algoritmo gauss naive_matriz y gauss separa_vectores dan la misma imagen\n\n");
	else printf("Algoritmo gauss naive_matriz y gauss separa_vectores dan distinta imagen\n\n");


	/*separa_vectores con vect_vertical secuencial y vect_horizontal_parallel*/
	computeTime = separa_vectores(&image, &imageAux,PARAL_HOR_SEQ_VERT);
	printf("separa_vectores con vect_vertical secuencial y vect_horizontal_parallel time: %0.9f seconds\n", computeTime);

	if (matrGaussImage == imageAux) printf("Algoritmo naive_matriz y gauss separa_vectores dan la misma imagen\n\n");
	else printf("Algoritmo gauss naive_matriz y gauss separa_vectores dan distinta imagen\n\n");


	/*separa_vectores con vect_vertical_paralle y vect_horizontal_parallel */
	computeTime = separa_vectores(&image, &imageAux,PARAL);
	printf("separa_vectores con vect_vertical_paralle y vect_horizontal_parallel time: %0.9f seconds\n", computeTime);

	if (matrGaussImage == imageAux) printf("Algoritmo gauss naive_matriz y gauss separa_vectores dan la misma imagen\n");
	else printf("Algoritmo gauss naive_matriz y gauss separa_vectores dan distinta imagen\n");


    QPixmap pixmap = pixmap.fromImage(imageAux);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

    view.show();
    return a.exec();
}
