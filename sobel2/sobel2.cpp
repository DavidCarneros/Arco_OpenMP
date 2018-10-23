/**********************************************************************************************
*
*       SERGIO GONZALEZ VELAZQUEZ y DAVID CARNEROS PRADO 
*
*   --> Implementacion:
*			
*			-Para hacer uso de la localidad de los datos del algoritmo utilizamos una matriz de 3x3
*			a la que llamamos 'local'. En ella se almacena el valor de los 6 píxeles que se reutilizan 
*			y los 3 nuevos.
*
*			-Las funciones 'cargarLocal' y 'desplazarLocal' se utilizan para manipular dicha matriz.
*			la funcion 'cargarLocal'se utiliza cada vez que se cambia de fila en el recorrido de la
*			imagen, puesto que en ese momento no se pueden reutilizar pixeles anteriores y tenemos que 
*			llenar la matriz local con el valor de los pixeles vecinos al pixel seleccionado. 
*			Por su parte, la funcíon 'desplazarLocal' se utiliza cada vez que se selecciona un pixel
*			de la misma fila pero siguiente columna para, descartar los 3 pixeles que ya no se utilizan
*			y cargar los 3 nuevos pixeles vecinos al nuevo pixel seleccionado.
*
*
*	
*	--> Conclusiones sobre la Tarea2:
*			-Usando la localidad se consigue reducir de manera considerable el tiempo del algoritmo basico. El tiempo
*			del algoritmo sobel que hace uso de la localidad es aproximadamente la mitad del tiempo del algoritmo basico.
*			Es un resultado logico, pues reutilizando el valor de pixeles estamos reduciendo considerablemente el numero
*			de accesos a memoria.
* 
*
*			-En lo que respecta a los algoritmos que usan los dos kernels, la diferencia de tiempos entre la version
*			secuencial y la version paralela es insignificante. 
*
*
*   --> Nota: Los tiempos de ejecucion utilizados para llegar a las conclusiones
*             han sido obtenidos ejecutando el programa en un ordenador con 2 nucleos.
*
*             En dicha ejecucion, se ha obtenido la siguiente salida:
*
*               -SobelBasico: 	 		0,208841042 segundos
*               -SobelParallel:  		0,297414780 segundos
*               -SobelLocal: 			0,119368309 segundos
*				-SobelLocalParallel: 	0,079807512 segundos
*               -SobelCompleto: 		0,139035615 segundos
*               -SobelCompletoParallel: 0,135005969 segundos
*
************************************************************************************************/


#include <QtGui/QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#define COLOUR_DEPTH 4

int weight[3][3] = 	{{ 1,  2,  1 },
					 { 0,  0,  0 },
					 { -1,  -2,  -1 }
					};

int kernel2[3][3] = {{ -1, 0, 1},
					 { -2, 0, 2 },
					 { -1, 0, 1}};


double SobelBasico(QImage *srcImage, QImage *dstImage) {
  double start_time = omp_get_wtime();
  int pixelValue, ii, jj, blue;

  for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      // Aplicamos el kernel weight[3][3] al pixel y su entorno
      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          for (int j = -1; j <= 1; j++) {
			blue = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            pixelValue += weight[i + 1][j + 1] * blue;	// En pixelValue se calcula el componente y del gradiente
          }
      }

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino
    }
  }
  return omp_get_wtime() - start_time;  
}

double SobelParallel(QImage *srcImage, QImage *dstImage) {
  double start_time = omp_get_wtime();
  int pixelValue, ii, jj, blue;

  #pragma omp parallel for private(jj,pixelValue,blue) schedule(dynamic,((srcImage->height())/omp_get_num_procs()))

  for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      // Aplicamos el kernel weight[3][3] al pixel y su entorno
      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          for (int j = -1; j <= 1; j++) {
			blue = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            pixelValue += weight[i + 1][j + 1] * blue;	// En pixelValue se calcula el componente y del gradiente
          }
      }

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
	  #pragma omp critical 
	  {
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino
      }
    }
  }
  return omp_get_wtime() - start_time;  
}


void cargarLocal(int local[][3] ,QImage *srcImage,int ii){
	for(int i=-1; i<=1;i++){
		for(int j=-1; j<= 1 ; j++){
			local[1+i][1+j]=qBlue(srcImage->pixel(1+j, ii+i));
		}
	}
}

void desplazarLocal(int local[][3],int ii,int jj,QImage *srcImage){

	local[0][0] = local[0][1];
	local[1][0] = local[1][1];
	local[2][0] = local[2][1];

	local[0][1] = local[0][2];
	local[1][1] = local[1][2];
	local[2][1] = local[2][2];

	local[0][2] = qBlue(srcImage->pixel(jj+2, ii-1));
	local[1][2] = qBlue(srcImage->pixel(jj+2, ii+0));
	local[2][2] = qBlue(srcImage->pixel(jj+2, ii+1));
}


double SobelLocal(QImage *srcImage, QImage *dstImage) {
  double start_time = omp_get_wtime();
  int pixelValue, ii, jj, blue;

  int local[3][3];
  memset(local,0,sizeof(int)*3*2);

  for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    
  	 cargarLocal(local,srcImage,ii); 
  	 /*Cada vez que hay un cambio de fila (ii aumenta su valor) tenemos que volver a
  	 a llenar la matriz local porque no se pueden reutilizar el valor de otros pixeles */ 

    for (jj = 1; jj < srcImage->width() - 1; jj++) {      
     // Aplicamos el kernel weight[3][3] al pixel y su entorno
      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3] y la matriz local[3][3]
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1]; //Reutilizamos el valor de los pixeles almacenados en local
            pixelValue += weight[i + 1][j + 1] * blue;	// En pixelValue se calcula el componente 'y' del gradiente
          }
      }

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

      if(jj!=srcImage->width()-2){ 
      /*La matriz local se desplaza mientras el pixel actual este en una fila anterior a la penultima*/
      	desplazarLocal(local,ii,jj,srcImage);
      }
      

    }
  }
  return omp_get_wtime() - start_time;  
}


double SobelLocalParallel(QImage *srcImage, QImage *dstImage) {
  double start_time = omp_get_wtime();
  int pixelValue, ii, jj, blue;

  int local[3][3];
  memset(local,0,sizeof(int)*3*2);

  #pragma omp parallel for private(pixelValue,blue,local,jj) schedule(dynamic,((srcImage->height())/omp_get_num_procs()))

  for (ii = 1; ii < srcImage->height() - 1; ii++) { 
    
  	 cargarLocal(local,srcImage,ii); 
    for (jj = 1; jj < srcImage->width() - 1; jj++) {

      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1];  
            pixelValue += weight[i + 1][j + 1] * blue;	
          }
      }

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	 
      
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	
      
      
      if(jj!=srcImage->width()-2){
      	desplazarLocal(local,ii,jj,srcImage);
      }
      

    }
  }
  return omp_get_wtime() - start_time;  
}

double SobelCompleto(QImage *srcImage, QImage *dstImage) {
  double start_time = omp_get_wtime();
  int pixelValue, ii, jj, blue,gx,gy;

  int local[3][3];
  memset(local,0,sizeof(int)*3*2);

  for (ii = 1; ii < srcImage->height() - 1; ii++) { 
    
  	cargarLocal(local,srcImage,ii); 

    for (jj = 1; jj < srcImage->width() - 1; jj++) {
     
      gy=0,gx=0;
      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1];  
			gy += weight[i+1][j+1]*blue;
			gx += kernel2[i+1][j+1]*blue;
			
 
          }
      }

      pixelValue += fabs(sqrt(pow(gy,2)+pow(gx,2)));

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

      if(jj!=srcImage->width()-2){
      	desplazarLocal(local,ii,jj,srcImage);
      }
      

    }
  }
  return omp_get_wtime() - start_time;  
}

double SobelCompletoParallel(QImage *srcImage, QImage *dstImage) {
  double start_time = omp_get_wtime();
  int pixelValue, ii, jj, blue,gx,gy;

  int local[3][3];
  memset(local,0,sizeof(int)*3*2);

  #pragma omp parallel for private(pixelValue,jj,blue,gx,gy,local) schedule(dynamic,((srcImage->height())/omp_get_num_procs()))

  for (ii = 1; ii < srcImage->height() - 1; ii++) {  	
    
  	cargarLocal(local,srcImage,ii); 
    for (jj = 1; jj < srcImage->width() - 1; jj++) {
      gy=0,gx=0;
      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1];  
			gy += weight[i+1][j+1]*blue;
			gx += kernel2[i+1][j+1]*blue;

          }
      }
      
      pixelValue += fabs(sqrt(pow(gy,2)+pow(gx,2)));

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

      if(jj!=srcImage->width()-2){
      	desplazarLocal(local,ii,jj,srcImage);
      }
      

    }
  }
  return omp_get_wtime() - start_time;  
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsView view(&scene);

    if (argc != 2) {printf("Vuelva a ejecutar. Uso: <ejecutable> <archivo imagen> \n"); return -1;} 
    QPixmap qp = QPixmap(argv[1]);
    if(qp.isNull()) { printf("no se ha encontrado la imagen\n"); return -1;}
	    
    QImage image = qp.toImage();
    QImage sobelImage(image);
    QImage sobelImageAux(image);
    QImage sobelCompletoImage(image);

    double computeTime = SobelBasico(&image, &sobelImage);
    printf("tiempo Sobel básico: %0.9f segundos\n", computeTime);

    computeTime = SobelParallel(&image,&sobelImageAux);
    printf("tiempo Sobel paralelo: %0.9f segundos\n", computeTime);

    /*  comparaciob sobel secuencial y sobel paralelo  */
	if (sobelImage == sobelImageAux) printf("Algoritmo sobel basico y sobel paralelo dan la misma imagen\n");
	else printf("Algoritmo sobel basico y sobel paralelo dan distinta imagen\n");

	printf("\n");
	/*  comparaciob sobel secuencial y sobel local  */

    computeTime = SobelLocal(&image,&sobelImageAux);
    printf("tiempo Sobel Local: %0.9f segundos\n", computeTime);

	if (sobelImage == sobelImageAux) printf("Algoritmo sobel basico y sobel local dan la misma imagen\n");
	else printf("Algoritmo sobel basico y sobel local dan distinta imagen\n");

	printf("\n");

	computeTime = SobelLocalParallel(&image,&sobelImageAux);
    printf("tiempo Sobel Local paralelo: %0.9f segundos\n", computeTime);


	if (sobelImage == sobelImageAux) printf("Algoritmo sobel basico y sobel local paralelo dan la misma imagen\n");
	else printf("Algoritmo sobel basico y sobel local paralelo dan distinta imagen\n");


	printf("\nSobel completo \n");

	computeTime = SobelCompleto(&image,&sobelCompletoImage);
    printf("tiempo Sobel completo: %0.9f segundos\n", computeTime);

    computeTime = SobelCompleto(&image,&sobelImageAux);
    printf("tiempo Sobel paralelo: %0.9f segundos\n", computeTime);


	if (sobelCompletoImage == sobelImageAux) printf("Algoritmo sobel completo y sobel completo paralelo dan la misma imagen\n");
	else printf("Algoritmo sobel completo y sobel completo paralelo dan distinta imagen\n");

    /*Visualizacion de la imagen resultante de aplicar los dos kernels */
    QPixmap pixmap = pixmap.fromImage(sobelCompletoImage);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

   view.show();
   return a.exec();
}
