# Natural language processing_FOMC Minutes

## Introduction
Deep learning models have shown remarkable success in medical image analysis tasks, including semantic segmentation.
This project aims to develop predictive models to label carotid artery areas 
of input sonography images which are obtained from EDA hospital.
I used the deep learning model ResNet18 Unet under Pytorch framework to do the analysis and not only got the amazing testing dice cofficient of 0.96505 but also ranked the top three among competitors from Statistics, Computer Science background.
[Kaggle Competition](https://www.kaggle.com/competitions/mia-hw4/leaderboard) (Ranking: 3/31, Testing Dice Cofficient: 0.96505)
![](/images/ranking.png "My Competition Ranking")

## Challenge
1. **Choosing between Last or Best parameter setting?**
   - After training each epoch, model parameters were logged and the best one was chosen in the condition that smallest validation loss. However, parameters from last epoch are not need to be the best durning the whole training process. So, how to determine whether training with more epochs or staying in the best in the current epoch is the trade-off between prediction accuracy and the time consumption.

2. **Try different combinations of hyperparameters?**
   - There are infinite combinations of hyperparameters, in my project I mainly focus on trying different combinations of learning rate and weight decay using grid search method. With the help of visualization provided by wandb, it is easier to compare and understand the performance of each combination comprehensively and quickly.

3. **Validation loss remain the same in the each training epoch**
   - I encounter the problem of the not declined validation loss. The predictions are identical, increasing the epochs may not improve the model's performance. Trying the different learning rate help to avoid getting stuck in a local minimum of the loss function. 

4. Trying to use binary entropy loss always feels like it could lead to better performance in binary classification tasks, but due to time constraints, this attempt was not successful, instead common cross entropy loss was used as a criterion for computing loss.

5. Because the characteristic of this set of images, training the model with more images (increasing from 240 to 270) allow the model to better capture the characteristics of the images.

6. Without the help of Cross Validation, it's impossible to know if there's a possibility of overfitting.
7. Due to the lack of data, data augmentation may help to enhance training result.




## Data Preparation
I define a class `FedMinScraper` aimed at extracting monthly US Federal Reserve minutes from the Federal Reserve website. It takes a list of dates as input, specifying the periods for which transcripts are to be extracted. The class utilizes multithreading for faster extraction and parsing of the transcripts. 

```python
class FedMinScraper(object):
    """
    The purpose of this class is to extract monthly US federal reserve minutes
    
    Parameters
    ----------
    dates: list('yyyy'|'yyyy-mm')
        List of strings/integers referencing dates for extraction
        Example:
        dates = [min_year] -> Extracts all transcripts for this year
        dates = [min_year,max_year] -> Extracts transactions for a set of years
        dates = ['2020-01'] -> Extracts transcripts for a single month/year

    nthreads: int
        Set of threads used for multiprocessing
        defaults to None

    Returns
    --------
    transcripts: txt files

    """

    url_parent = r"https://www.federalreserve.gov/monetarypolicy/"
    url_current = r"fomccalendars.htm"

    # historical transcripts are stored differently
    url_historical = r"fomchistorical{}.htm"
    # each transcript has a unique address, gathered from url_current or url_historical
    url_transcript = r"fomcminutes{}.htm"
    href_regex = re.compile("(?i)/fomc[/]?minutes[/]?\d{8}.htm")

    def __init__(self, dates, nthreads=5, save_path=None):

        # make sure user has given list with strings
        if not isinstance(dates, list):
            raise TypeError("dates should be a list of yyyy or yyyymm str/int")

        elif not all([bool(re.search(r"^\d{4}$|^\d{6}$", str(d))) for d in dates]):
            raise ValueError("dates should be in a yyyy or yyyymm format")

        self.dates = dates
        self.nthreads = nthreads
        self.save_path = save_path

        self.ndates = len(dates)
        self.years = [int(d[:4]) for d in dates]
        self.min_year, self.max_year = min(self.years), max(self.years)
        self.transcript_dates = []
        self.transcripts = {}
        self.historical_date = None

        self._get_transcript_dates()

        self.start_threading()

        if save_path:
            self.save_transcript()

    def _get_transcript_dates(self):
        """
        Extract all links for
        """

        r = requests.get(urljoin(FedMinScraper.url_parent, FedMinScraper.url_current))
        soup = BeautifulSoup(r.text, "lxml")
        # dates are given by yyyymmdd

        tdates = soup.findAll("a", href=self.href_regex)
        tdates = [re.search(r"\d{8}", str(t))[0] for t in tdates]
        self.historical_date = int(min(tdates)[:4])
        # find minimum year

        # extract all of these and filter
        # tdates can only be applied to /fomcminutes
        # historical dates need to be applied to federalreserve.gov

        if self.min_year < self.historical_date:
            # just append the years i'm interested in
            for y in range(self.min_year, self.historical_date + 1):

                r = requests.get(
                    urljoin(
                        FedMinScraper.url_parent, FedMinScraper.url_historical.format(y)
                    )
                )
                soup = BeautifulSoup(r.text, parser="lxml")
                hdates = soup.find_all("a", href=self.href_regex)
                tdates.extend([re.search(r"\d{8}", str(t))[0] for t in hdates])

        self.transcript_dates = tdates

    def get_transcript(self, transcript_date):

        transcript_url = urljoin(
            FedMinScraper.url_parent,
            FedMinScraper.url_transcript.format(transcript_date),
        )
        r = requests.get(transcript_url)

        if not r.ok:
            transcript_url = urljoin(
                FedMinScraper.url_parent.replace("/monetarypolicy", ""),
                r"fomc/minutes/{}.htm".format(transcript_date),
            )
            r = requests.get(transcript_url)

        soup = BeautifulSoup(r.content, "lxml")
        main_text = soup.findAll(name="p")

        clean_main_text = "\n\n".join(t.text.strip() for t in main_text)

        # reduce double spaces to one
        clean_main_text = re.sub(r"  ", r" ", clean_main_text)

        self.transcripts[transcript_date] = clean_main_text

    def start_threading(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nthreads) as executor:
            executor.map(self.get_transcript, self.transcript_dates)

    def save_transcript(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for fname, txt in self.transcripts.items():
            with open(
                os.path.join(self.save_path, fname + ".txt"), "w", encoding="utf-8"
            ) as o:
                o.write(txt)
                o.close()
```

The class uses BeautifulSoup for web scraping, fetching the URLs of the transcripts based on the provided dates. It then extracts the main text content from the fetched transcripts, cleans the text, and stores the cleaned transcripts in a dictionary with keys representing the respective dates. 

The extracted transcripts can also be saved to a specified directory if provided. The class verifies the input dates and ensures they are in the correct format before proceeding with extraction. It distinguishes between current and historical transcripts based on the provided dates and constructs the appropriate URLs for extraction.

Overall, this class provides a convenient way to automate the retrieval and storage of Federal Reserve meeting transcripts for analysis or archival purposes.

After having the FOMC minutes text documents, I truncated the documents discarding the unnecessary information such as the list of attendants presented in the begining of the FOMC minute. 
![](/images/minute.png "Attendance info should be truncated before NLP")

## EDA

### Descriptive Statistics

![](/images/EDA.png "minute information")

### Paragraphs and words overtime

![](/images/year.png "paragraph and word over time")

## Model Building
FCN model was first being used, however it's testing dice accuracy not surpassing 0.8 threshold. Thus, I turned into Unet model proposed by  . Especially, ResNet18 was used as encoder block and the decoder block follwed by original paper. Pre-trained model was deployed because of the small amount of data. Fine-tuning process was displayed visualizing in the wandb api dashboard. After tring the various combination of hyperparameter such as learning rate and weight decay etc. I used the set of parameter listed below as my hyperparamter setting. Include code snippets or references to notebooks where the model building process is documented.

## Metrics

Two commonly used metrics to evaluate the performance of segmentation algorithms are Dice Coefficient and Intersection over Union (IoU).

### Dice Coefficient

The Dice Coefficient, also known as the F1 Score, is a measure of the similarity between two sets. In the context of image segmentation, it is used to quantify the agreement between the predicted segmentation and the ground truth.

The formula for Dice Coefficient is given by:

$$ Dice = \frac{2 \times |X \cap Y|}{|X| + |Y|} $$

where:
- $X$ is the set of pixels in the predicted segmentation,
- $Y$ is the set of pixels in the ground truth,
- $|\cdot|$ denotes the cardinality of a set (i.e., the number of elements).

Dice Coefficient ranges from 0 to 1, where 1 indicates a perfect overlap between the predicted and ground truth segmentations.

### Intersection over Union (IoU)

IoU, also known as the Jaccard Index, is another widely used metric for segmentation evaluation. It measures the ratio of the intersection area to the union area between the predicted and ground truth segmentations.

The formula for IoU is given by:

$$ IoU = \frac{|X \cap Y|}{|X \cup Y|} $$

where:
- $X$ is the set of pixels in the predicted segmentation,
- $Y$ is the set of pixels in the ground truth.

Similar to Dice Coefficient, IoU ranges from 0 to 1, with 1 indicating a perfect overlap.

![](https://www.mathworks.com/help/vision/ref/jaccard.png)

### Interpretation

- **High Values**: A higher Dice Coefficient or IoU indicates better segmentation performance, as it signifies a greater overlap between the predicted and ground truth regions.

- **Low Values**: Lower values suggest poor segmentation accuracy, indicating a mismatch between the predicted and ground truth segmentations.

### Implementation

Dice coefficient and IoU can be calculated by confusion matrix. Therefore, the initial step is to build an confusion matrix from scratch.

**Algorithm: Building Confusion Matrix $M$**

**Input:**
- a: Target labels tensor
- b: Predicted labels tensor
- num_classes: Number of classes

**Procedure:**
1. Initialize the confusion matrix (self.mat) if it is not already created:
   - Create a square matrix of zeros with shape (num_classes, num_classes) and dtype=torch.int64.
   - Place the matrix on the same device as the input tensor `a`.

2. Update the confusion matrix using the update method:

   a. Check for valid class indices:
      - Create a boolean mask k, where elements are True if a is in the range [0, num_classes) and False otherwise.
      
   b. Calculate indices for updating the confusion matrix:
      - Convert the valid elements of `a` and `b` to torch.int64 and calculate the indices using the formula n * a[k] + b[k], where n is the number of classes.
         - **We represent class-i pixels classify to class-j as $\to n * i + j$**
      - Increment the corresponding elements in the confusion matrix using torch.bincount.

3. Compute segmentation metrics using the compute method:
   - Convert the confusion matrix to a float tensor h.
   - Extract correct predictions along the diagonal of the matrix.
   - Compute metrics from $M$
      - `acc` $\to M_{ii}/M_{i\cdot}$
      - `global_acc` $\to sum(M_{ii})/M_{\cdot\cdot}$
      - `dice` $\to \frac{2M_{ii}}{M_{i\cdot}+M_{\cdot i}}$
      - `iou` $\to \frac{M_{ii}}{M_{i\cdot}+M_{\cdot i}-M_{ii}}$

**Output:**
- The confusion matrix is updated and segmentation metrics are computed.
- The $(i, j)-$ terms of the $M$ represents class-i pixels classify to class-j

Note: In practice, we often omit the metrics from the background!!
