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


double separa_vectores(QImage* image, QImage* result, int flag){

  double start_time = omp_get_wtime();
   
  switch(flag){
    case SEQ:

    break;

    case PARAL:
    break;

    case SEQ_HOR_PARAL_VERT:

    break;

    case PARAL_HOR_SEQ_VERT:

    break;
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
    if(qp.isNull()) { printf("image not found\n"); return -1;}
    
    QImage image = qp.toImage();
    
    alto = image.height(); ancho = image.width();
    
    QImage matrGaussImage(image);
    
    double computeTime = naive_matriz(&image, &matrGaussImage);
    printf("naive_matriz time: %0.9f seconds\n", computeTime);
    
    QPixmap pixmap = pixmap.fromImage(matrGaussImage);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

    view.show();
    return a.exec();
}
