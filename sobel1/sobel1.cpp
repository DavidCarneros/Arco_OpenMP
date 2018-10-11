#include <QtGui/QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>

#define COLOUR_DEPTH 4

int weight[3][3] = 	{{ 1,  2,  1 },
					 { 0,  0,  0 },
					 { -1,  -2,  -1 }
					};

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
    
    double computeTime = SobelBasico(&image, &sobelImage);
    printf("tiempo Sobel básico: %0.9f segundos\n", computeTime);
   
    QPixmap pixmap = pixmap.fromImage(sobelImage);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

    view.show();
    return a.exec();
}
