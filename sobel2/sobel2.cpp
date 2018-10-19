/******************************************************************
*
*       SERGIO GONZALEZ VELAZQUEZ y DAVID CARNEROS PRADO 
*
*   tiempo Sobel básico: 0,128611388 segundos
*   tiempo Sobel Paralelo sin 'schedule': 0,175052620 segundos
*
*   schedule static 6 --> 0,239515744 segundos
*   schedule dynamic 6 -->  0,204192400 segundos
*   schedule static altura/nºprocesadores --> 0,203713559 segundos
*   shcedule dynamic alutra/nºprocesadores --> 0,162446580 segundos
* 
*   La funcion paralelizada que mejor tiempo tiene en la
*   la mayoria de las ejecuciones es en la que utilizamos la clausua 
*   schedule asignando las iteraciones a las hebras con 
*   'dynamic (altura/nº de procesadores)'. 
*
*   No obstante, la funcion paralelizada no reduce el tiempo de ejecucion
*   de la version secuencial, ya que existe una seccion critica y las hebras 
*   deben sincronizarse.
****************************************************************/


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

int kernel2[3][3] = {{-1,0,1},
					 {-2,0,2},
					 {-1,0,1}};


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

void recargarLocal(int local[][3],int ii,int jj,QImage *srcImage){

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
    
  	 cargarLocal(local,srcImage,ii); // Recargar la matriz local 
  	 // Cada vez que se cambia de fila se recagarga la matriz local. 

    for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
     // Aplicamos el kernel weight[3][3] al pixel y su entorno

      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1];  //qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            pixelValue += weight[i + 1][j + 1] * blue;	// En pixelValue se calcula el componente y del gradiente
          }
      }

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

      if(jj!=srcImage->width()-2){
      	recargarLocal(local,ii,jj,srcImage);
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

  #pragma omp parallel for private(pixelValue,local,blue,jj) schedule(dynamic,((srcImage->height())/omp_get_num_procs()))

  for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    
  	 cargarLocal(local,srcImage,ii); // Recargar la matriz local 
  	 // Cada vez que se cambia de fila se recagarga la matriz local. 

    for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
     // Aplicamos el kernel weight[3][3] al pixel y su entorno

      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1];  //qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            pixelValue += weight[i + 1][j + 1] * blue;	// En pixelValue se calcula el componente y del gradiente
          }
      }

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

      if(jj!=srcImage->width()-2){
      	recargarLocal(local,ii,jj,srcImage);
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

  for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    
  	 cargarLocal(local,srcImage,ii); // Recargar la matriz local 
  	 // Cada vez que se cambia de fila se recagarga la matriz local. 

    for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
     // Aplicamos el kernel weight[3][3] al pixel y su entorno
      gy=0,gx=0;
      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1];  //qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
			gy += weight[i+1][j+1]*blue;
			gx += kernel2[i+1][j+1]*blue;
			
           	// En pixelValue se calcula el componente y del gradiente
          }
      }

      pixelValue += fabs(sqrt(pow(gy,2)+pow(gx,2)));

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

      if(jj!=srcImage->width()-2){
      	recargarLocal(local,ii,jj,srcImage);
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

  for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    
  	 cargarLocal(local,srcImage,ii); // Recargar la matriz local 
  	 // Cada vez que se cambia de fila se recagarga la matriz local. 

    for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
     // Aplicamos el kernel weight[3][3] al pixel y su entorno
      gy=0,gx=0;
      pixelValue = 0;
      for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          for (int j = -1; j <= 1; j++) {
			blue = local[i+1][j+1];  //qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
			gy += weight[i+1][j+1]*blue;
			gx += kernel2[i+1][j+1]*blue;
			
           	// En pixelValue se calcula el componente y del gradiente
          }
      }
      
      pixelValue += fabs(sqrt(pow(gy,2)+pow(gx,2)));

      if (pixelValue > 255) pixelValue = 255;
      if (pixelValue < 0) pixelValue = 0;
	
      dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

      if(jj!=srcImage->width()-2){
      	recargarLocal(local,ii,jj,srcImage);
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

    QPixmap pixmap = pixmap.fromImage(sobelCompletoImage);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

   view.show();
   return a.exec();
}
