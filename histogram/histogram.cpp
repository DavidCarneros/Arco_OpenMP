/*******************************************************/
/*                                                     */
/* SERGIO GONZALEZ VELAZQUEZ y DAVID CARNEROS PRADO    */
/*                                                     */
/*******************************************************/


#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>

#define COLOUR_DEPTH 4

omp_lock_t lock[256];

double computeGraySequential(QImage *image) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);
    *rgbpixel = QColor(gray, gray, gray).rgba();
  }
  return omp_get_wtime() - start_time;
}

double computeGrayParallel(QImage *image) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

#pragma omp parallel for
  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);
    *rgbpixel = QColor(gray, gray, gray).rgba();
  }
  return omp_get_wtime() - start_time;
}

double computeGrayScanline(QImage *image) {
  double start_time = omp_get_wtime();
  int alto = image->height(); int ancho = image->width();
  int jj, gray; uchar* scan; QRgb* rgbpixel;
  for (int ii = 0; ii < alto; ii++) {

    scan = image->scanLine(ii);
    for (jj = 0; jj < ancho; jj++) {

      rgbpixel = reinterpret_cast<QRgb*>(scan + jj * COLOUR_DEPTH);
      gray = qGray(*rgbpixel);
      *rgbpixel = QColor(gray, gray, gray).rgba();
    }
  }
  return omp_get_wtime() - start_time;
}

/************ FUNCION AÑADIDA ***************/

/* Hemos paralelizado el primer "for", ya que es de reparto de trabajo.
El segundo "for" sin embargo, no es de reparto de trabajo, cuando entra una hebra
lo ejecuta secuencialmente de manera completa. Para ello hemos utilizado la directiva
pragma omp parallel for, y hemos declarado como variables privadas las que se utilizan
dentro del segundo bucle (jj,rgbpixel,gray y scan).
Para proteger como seccion critica la instrucion "scan = image->scanLine(ii);" Hemos
utlizado la directiva critical ya que no podiamos utilizar ni atomic ni reduction. */

double computeGrayScanlineParallel(QImage *image) {
  double start_time = omp_get_wtime();
  int alto = image->height(); int ancho = image->width();
  int jj, gray; uchar* scan; QRgb* rgbpixel;

  #pragma omp parallel for private(jj,rgbpixel,gray,scan)
  for (int ii = 0; ii < alto; ii++) {

    #pragma omp critical
    {
    scan = image->scanLine(ii);
    }

    for (jj = 0; jj < ancho; jj++) {

      rgbpixel = reinterpret_cast<QRgb*>(scan + jj * COLOUR_DEPTH);
      gray = qGray(*rgbpixel);
      *rgbpixel = QColor(gray, gray, gray).rgba();
    }
  }
  return omp_get_wtime() - start_time;
}


/**************************************************/


double computeHistogramSecuencial(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);
    histgr[gray]++;
  }
  return omp_get_wtime() - start_time;
}

double computeHistogramCritical(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  #pragma parallel for private(QRgb,gray)

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

   #pragma omp critical
    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}

double computeHistogramReduction(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  #pragma parallel for private(QRgb,gray) reduction(+:histgr)

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}

double computeHistogramAtomic(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  #pragma parallel for private(QRgb,gray)

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

   #pragma omp atomic
    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}


void createLocks(){
	for(int i=0;i<256;i++){
		omp_init_lock(&lock[i]);
	}
}

void destroyLocks(){
	for(int i=0;i<256;i++){
		omp_destroy_lock(&lock[i]);
	}
}

double computeHistogramLock(QImage *image,int histgr[]) {

  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();
  createLocks();

  #pragma parallel for private(QRgb,gray)

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);
    omp_set_lock(&lock[gray]);
    histgr[gray]++;
    omp_unset_lock(&lock[gray]);
  }

  destroyLocks();

  return omp_get_wtime() - start_time;
}

bool compararHist(int histgr[],int histgrAux[]){
	int i=0;
	bool igual = true;

	for (int i=0;i<256 && igual;i++){
		igual = (histgr[i]==histgrAux[i]);
	}

	return igual;
}



int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsView view(&scene);
    QPixmap qp = QPixmap("test_1080p.bmp"); // ("c:\\test_1080p.bmp");

    int histgr[256];
    int histgrAux[256];

    memset(histgr,0,sizeof(histgr));
    memset(histgrAux,0,sizeof(histgrAux));

    if(qp.isNull())
    {
        printf("image not found\n");
		return -1;
    }

    QImage image = qp.toImage();

    QImage seqImage(image);

    double computeTime = computeHistogramSecuencial(&image,histgr);
    printf("sequential time: %0.9f seconds\n", computeTime);

    computeTime = computeHistogramCritical(&image,histgrAux);
    printf("Critical time: %0.9f seconds\n", computeTime);

    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y critial son iguales \n");
    else printf("Algoritmo secuencial y critical son diferentes \n"); 

	memset(histgrAux,0,sizeof(histgrAux));
    computeTime = computeHistogramAtomic(&image,histgrAux);
    printf("Atomic time: %0.9f seconds\n", computeTime);

    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y Atomic son iguales \n");
    else printf("Algoritmo secuencial y Atomic son diferentes \n"); 


	memset(histgrAux,0,sizeof(histgrAux));
    computeTime = computeHistogramReduction(&image,histgrAux);
    printf("Reduction time: %0.9f seconds\n", computeTime);

    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y reduction son iguales \n");
    else printf("Algoritmo secuencial y reduction son diferentes \n"); 

	memset(histgrAux,0,sizeof(histgrAux));
    computeTime = computeHistogramLock(&image,histgrAux);
    printf("Locks time: %0.9f seconds\n", computeTime);

    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y Locks son iguales \n");
    else printf("Algoritmo secuencial y Locks son diferentes \n"); 

   /*
   QPixmap pixmap = pixmap.fromImage(auxImage);
   QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
   scene.addItem(item);

   view.show();



    return a.exec();
    */
}
