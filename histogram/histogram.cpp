/******************************************************************************************************************
*
* 			SERGIO GONZALEZ VELAZQUEZ y DAVID CARNEROS PRADO 
*
*   --> Implementacion:
*   
*      -En las funciones paralelizadas no hemos establecido explicitamente ninguna variable
*       privada porque se declaran implicitamente la variable 'gray' al estar declarada
*       dentro de la seccion paralela.       
*       
*       -Utilizamos dos vectores 'histograma', uno que se rellena unicamente en la version secuencial
*       y otro que sera utilizado en las versiones paralelizadas. Esto nos permite que, despues de obtener
*       el resultado de una version paralelizada, comprobar que el resultado es el correcto comparandolo 
*        con el resultado de la version secuencial. 
*        Es importante, poner a 0 las componentes de este vector antes de la ejecucion de cada funcion paralelizada, 
*        pues al ser el mismo vector, de no 'reiniciarlo' se solaparian los resultados de las distintas funciones.
*
*       -En cuanto al establecimiento del numero de hebras en las secciones paralelas,
*       hemos dejado que lo haga el sistema en tiempo de ejecucion, ya que asigna tipicamente
*        tantas hebras como nucleos haya disponibles.
*
*
*
*   --> Conclusiones: 
*
*       -Utilizar la directiva 'critical' para gestionar la seccion critica
*        de la version paralelizada ha hecho que la funcion se ejecute mas lenta 
*        que la version secuencial. Este resultado era de esperar, pues hay que 
*        tener en cuenta que critical es sincroniza las hebras de forma que una hebra 
*        tiene que esperar para incrementar el vector histogram hasta 
*        que otra hebra haya terminado. Estas esperas implican una perdida de rendimiento
*        que se traduce en un mayor tiempo de ejecucion..
*
*        -Podemos comprobar que la seccion critica creada con la directiva 
*        'atomic' es mas ligera que la creada con 'critical', pues el tiempo 
*         de ejecución de la función que utiliza 'atomic' es aproximadamente
*         10 veces menor que el de la función que utiliza 'critical'.
*
*         No obstante, paralelizar la funcion protegiendo el acceso al histograma
*         con 'atomic' no ha hecho que sea mas rapida que la version secuencial por 
*         los motivos explicados anteriormente. Asi pues, la version con 'atomic' tarda
*         aproximadamente el doble de tiempo en ejecutarse que la version secuencial.
*
*        -La forma de mas eficiente de paralelizar la version secuencial es utilizando la clausula 
*        'reduction', con la que mejoramos el tiempo de ejecucion en, aproximadamente, un 500%.
*        
*
*        -Por ultimo, utilizando locks de bajo nivel tampoco hemos conseguido mejorar el tiempo
*         el tiempo de ejecucion de la version secuencial. Sin embargo, esta forma de paralelizar
*         ha resultado mas eficiente que utilizando la directiva 'critical'
*
*   --> Nota: Los tiempos de ejecucion utilizados para llegar a las conclusiones
*             han sido obtenidos ejecutando el programa en un ordenador con 2 nucleos.
*             Dichos resultados son orientativos, pues en diferentes ejecuciones se obtienen 
*             diferentes valores.
*             En dicha ejecucion, se ha obtenido la siguiente salida:
*
*               -sequential time: 0,013672096 seconds
*               -critical time: 0,274625684 seconds
*               -atomic time: 0,027022785 seconds
*               -reduction time: 0,002560306 seconds
*               -locks time: 0,060401263 seconds
*
**********************************************************************************************************************/


#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>

#define COLOUR_DEPTH 4
#define LEVELS 256 //Niveles de gris

omp_lock_t lock[256];


/* -------- VERSION SECUENCIAL -------- */

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


/* -------- VERSION PARALELA CON CRITICAL -------- */

double computeHistogramCritical(QImage *image,int *histgr) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  memset(histgr,0,sizeof(int) * LEVELS);  //Se inicializan los componentes a 0

  #pragma omp parallel for 

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

   #pragma omp critical
    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}


/* -------- VERSION PARALELA CON ATOMIC -------- */

double computeHistogramAtomic(QImage *image,int *histgr) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();
  memset(histgr,0,sizeof(int) * LEVELS);    //Se inicializan los componentes a 0

  #pragma omp parallel for 

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

   #pragma omp atomic
    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}

/* -------- VERSION PARALELA CON REDUCTION-------- */
double computeHistogramReduction(QImage *image,int *histgr) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();
  memset(histgr,0,sizeof(int) * LEVELS);    //Se inicializan los componentes a 0

  #pragma omp parallel for reduction(+:histgr[:256])

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}




/* -------- VERSION PARALELA CON LOCKS DE BAJO NIVEL------- */
void createLocks(){
	for(int i=0;i<LEVELS;i++){
		omp_init_lock(&lock[i]);
	}
}

void destroyLocks(){
	for(int i=0;i<LEVELS;i++){
		omp_destroy_lock(&lock[i]);
	}
}

double computeHistogramLock(QImage *image,int histgr[]) {

  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();
  memset(histgr,0,sizeof(int) * LEVELS);  //Se inicializan los componentes a 0
  createLocks();

  #pragma omp parallel for 

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

    omp_set_lock(&lock[gray]);    // Entrada a la seccion critica
    histgr[gray]++;				        // Seccion critica
    omp_unset_lock(&lock[gray]);  // Salida de la seccion critica
  }

  destroyLocks();		//Destruimos los cerrojos para liberar memoria

  return omp_get_wtime() - start_time;
}


bool compararHist(int histgr[],int histgrAux[]){

	bool igual = true;

	for (int i=0;i<LEVELS && igual;i++){
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

  int histgr[LEVELS];
  int histgrAux[LEVELS];

  memset(histgr,0,sizeof(histgr));		//Se inicializan los componentes a 0
  memset(histgrAux,0,sizeof(histgrAux));

  if(qp.isNull())
  {
    printf("image not found\n");
		return -1;
  }

  QImage image = qp.toImage();
  QImage seqImage(image);

  double computeTime = computeHistogramSecuencial(&image,histgr);
  printf("sequential time: %0.9f seconds\n\n", computeTime);

  computeTime = computeHistogramCritical(&image,histgrAux);
  printf("critical time: %0.9f seconds\n", computeTime);
  if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'critical' dan el mismo histograma \n\n");
  else printf("Algoritmo secuencial y paralelo con 'critical' NO dan el mismo histograma \n\n");


  computeTime = computeHistogramAtomic(&image,histgrAux);
  printf("Atomic time: %0.9f seconds\n", computeTime);
  if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'atomic' dan el mismo histograma \n\n");
  else printf("Algoritmo secuencial y paralelo con 'atomic' NO dan el mismo histograma \n\n");


  computeTime = computeHistogramReduction(&image,histgrAux);
  printf("Reduction time: %0.9f seconds\n", computeTime);
  if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'reduction' dan el mismo histograma \n\n");
  else printf("Algoritmo secuencial y paralelo con 'reduction' NO dan el mismo histograma \n\n");

  

  computeTime = computeHistogramLock(&image,histgrAux);
  printf("Locks time: %0.9f seconds\n", computeTime);
  if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'locks' dan el mismo histograma \n\n");
  else printf("Algoritmo secuencial y paralelo con 'locks' NO dan el mismo histograma \n\n");


    
}
