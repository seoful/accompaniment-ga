# Accompaniment generation genetic algorithm

## Manual

I used Python 1.10 to code the task.

In order the program to work you need to install ```mido``` and ```music21``` libraries using ```pip install``` command. Also, you should put the MIDI file named ```input.mid``` in the same folder as the ```.py``` program file. After running the algorithm, ```output.mid``` will be generated in the same folder.

## Key detection
For detecting melody key I used ```music21``` library.
Input 1 - Dm
Input 2 - F
Input 3 - Em
## Genetic algorithm

Everything connected to a genetic algorithm, in my program is in class ```GA```. The accompaniment (list of chords) is a chromosome and a chord itself is a gene.

### Chords pool generation

While initializing ```GA``` class, I create the array of chords that can be used in a certain key. I'm doing it by iterating through all  notes with MIDI codes [21,108], and for each I check whether the scale contains that note. If yes, I count the step of the note in scale and add all possible chords for this step to the chord pool for current note.

### Selection

For selection I use truncation. I pick some percentage (that is a parameter) of a population with the best scores of fitness function and use them for crossover.

### Crossover

From selected accompaniments ```crossover_rate * population_size``` times random two accompaniments are chosen for breeding.

For this two accompaniments, I iterate through chords and with 0.5 probability swap respective chords (genes) between parents. So, two new offsprings are spawned and added to the population.

### Mutation

Each offspring goes through mutation stage, where every chord (gene) with ```gene_mutation_probability``` is replaced by random chord from the chord pool.

### Fitness function

Fitness function consists of 8 metrics that are calculated and summed up each multiplied by respective emperically chosen coefficients. In these metrics thera are both "awards" that give points and penalties that subtract some points from the tottal scores.

#### 1. Consonance score

For each note in each chord I gain some score for being consonant to the melody note played at that beat. For perferc consonances I give more points than for imperfect ones. If in a chord there is a dissonant note, I add 0 points for all notes in this chord without consideration of their consonancy. Consonant and dissonant intervals wew taken from [Wikipedia](https://en.wikipedia.org/wiki/Consonance_and_dissonance).

#### 2. Dissonance penalty

Penalty for chords containing dissonant notes to the respective melody note.

#### 3. Progressions score

I count the number of subarrays of chords in accompaniment that creates a chord progression. I check for three progressions: 50s progression (I–vi–IV–V with and starting point); I–V–vi–IV with any starting point, and ii–V–I one.

#### 4. Chord repetitions penalty

Counts the number of subarrays of ```window_size``` length that contains equal chords and penalizes the accompaniment for being so boring.

#### 5. Note - accompaniment pitch difference penalty

For accompaniment being consistent and not have the big octaves range, the average pitch of all notes is calculated. Then, the program finds the number of notes that are further away from that average than some ```acceptance_range``` and penalizes the accompaniment.

#### 6. Neighbors chords offset score

While creating fitness function, I came across the problem that often there were accompaniments with neighbor chords that lie too far away which makes it sound not consistent. So, this metric was added. It counts number of neighbor chords pairs that maximum difference of notes lies in some ```acceptance_range```.

#### 7. Melody - accompaniment pitch difference penalty

In order to create accompaniment in other octaves than the melody is I calculate the average pitch of melody and accompaniment. Then, I penalize the accompaniment if the difference between the averages is far from ```ideal_distance```. So, the GA prefers accompaniments that are in average in lower octaves than the melody.

#### 8. Beats-without-melody-notes accompaniment soundness penalty

There are beats on which there are no melody notes played. So, the algorithm cannot check the connsonancy and dissonancy. Thus, it creates something bad on this beats. I decided that it would sound better if I repeat the previous chord in such cases. So, this metric calculates number of such "melody gaps" where there is no chord repetition and penalizes accompaniment for it.

## Results

There are three ```input_.mid``` and relevant ```output_.mid``` files.