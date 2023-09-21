import copy

import mido
import music21
import random


# Class that represents the chord. It stores the root note of the chord, offsets from it and all the notes of th chord.
class Chord:
    def __init__(self, main_note: int, offsets: list[int]):
        self.main_note = main_note
        self.offsets = offsets
        self.notes = [main_note + offsets[i] for i in range(len(offsets))]

    def validate(self) -> bool:
        for note in self.notes:
            if note > 108:
                return False
        return True


# Holder for offsets of different types of chords.
class ChordsOffsets:
    major_triad_offsets = [0, 4, 7]
    major_first_inverse_offsets = [0, 4, 7]
    major_second_inverse_offsets = [12, 16, 7]
    minor_triad_offsets = [0, 3, 7]
    minor_first_inverse_offsets = [12, 3, 7]
    minor_second_inverse_offsets = [12, 15, 7]
    diminished_offsets = [0, 3, 6]
    sus2_offsets = [0, 2, 7]
    sus4_offsets = [0, 5, 7]


# Class that is responsible for everything with genetic algorithm
class GA:
    def __init__(self, melody_notes: list[int], melody_core_notes, melody_key: music21.key.Key, beats_number):
        note_sum = 0
        note_count = 0
        for note in melody_notes:
            note_sum += note
            note_count += 1
        self.average_note = note_sum / note_count

        self.core_melody_notes = melody_core_notes
        self.tonic = melody_key.tonic.midi
        self.scale = melody_key.mode
        self.chords_number = beats_number
        self.chords_pool = []

        # Initializing steps offsets for the given scale
        if self.scale == 'major':
            self.steps_offsets = [0, 2, 4, 5, 7, 9, 11]
        else:
            self.steps_offsets = [0, 2, 3, 5, 7, 8, 10]

        self._init_chords_pool()

    def _init_chords_pool(self):

        # creating sets of tonic scale steps (decreased by 1 to work index from 0)
        # for which different types of chords can be created
        if self.scale == 'major':
            major_steps = {0, 3, 4}
            minor_steps = {1, 2, 5}
            diminished_steps = {6}
            sus2_steps = {0, 1, 3, 4, 5}
            sus4_steps = {0, 1, 2, 4, 5}
        else:
            major_steps = {2, 5, 6}
            minor_steps = {0, 3, 4}
            diminished_steps = {1}
            sus2_steps = {0, 2, 3, 5, 6}
            sus4_steps = {0, 2, 3, 4, 6}

        # Iterating through all of notes and for each creating possible types of chords fitting the key of melody
        for note in range(21, 109):
            note_offset_from_tonic = abs(self.tonic - note) % 12
            if note_offset_from_tonic in self.steps_offsets:
                step = self.steps_offsets.index(note_offset_from_tonic)

                def add_chord_to_pool(offsets: list[int]):
                    chord = Chord(note, offsets)
                    if chord.validate():
                        self.chords_pool.append(chord)

                if step in major_steps:
                    add_chord_to_pool(ChordsOffsets.major_triad_offsets)
                    add_chord_to_pool(ChordsOffsets.major_first_inverse_offsets)
                    add_chord_to_pool(ChordsOffsets.major_second_inverse_offsets)
                if step in minor_steps:
                    add_chord_to_pool(ChordsOffsets.minor_triad_offsets)
                    add_chord_to_pool(ChordsOffsets.minor_first_inverse_offsets)
                    add_chord_to_pool(ChordsOffsets.minor_second_inverse_offsets)
                if step in diminished_steps:
                    add_chord_to_pool(ChordsOffsets.diminished_offsets)
                if step in sus2_steps:
                    add_chord_to_pool(ChordsOffsets.sus2_offsets)
                if step in sus4_steps:
                    add_chord_to_pool(ChordsOffsets.sus4_offsets)

    # Mutation: changing all genes to random each with given probability
    def mutate(self, chromosome: list[Chord], gene_mutation_probability: float) -> list[Chord]:
        for i in range(len(chromosome)):
            mutate = random.uniform(0, 1) < gene_mutation_probability
            if mutate:
                chromosome[i] = random.sample(self.chords_pool, k=1)[0]
        return chromosome

    # Crossover: exchanging i-th genes between two parents with 0.5 probability.
    def crossover(self, parent_1: list[Chord], parent_2: list[Chord]) -> (list[Chord], list[Chord]):

        offspring_1 = []
        offspring_2 = []
        for i in range(len(parent_1)):
            rand = random.randint(0, 1)
            offspring_1.append([parent_1[i], parent_2[i]][rand])
            offspring_2.append([parent_1[i], parent_2[i]][1 if rand == 0 else 0])
        return offspring_1, offspring_2

    # Function to run the genetic algorithm and get the output
    def run(self, population_size: int, iterations: int, selection_rate: float,
            crossover_rate: float,
            gene_mutation_probability: float) -> list[Chord]:
        population = [random.choices(population=self.chords_pool, k=self.chords_number) for _ in
                      range(population_size)]

        population = sorted(population, key=lambda elem: self.get_fitness(elem), reverse=True)
        for iteration in range(iterations):
            print("Iteration " + str(iteration) + " - Fitness: " + str(self.get_fitness(population[0])))

            # Selection: truncating the best
            population_after_selection_size = round(len(population) * selection_rate)
            crossover_number = round(len(population) * crossover_rate)

            population_after_selection = population[:population_after_selection_size]

            # Crossover + mutation
            new_population = []
            for _ in range(crossover_number):
                parents = random.sample(population_after_selection, k=2)
                offspring_1, offspring_2 = self.crossover(parents[0], parents[1])
                new_population.append(self.mutate(offspring_1, gene_mutation_probability))
                new_population.append(self.mutate(offspring_2, gene_mutation_probability))
            new_population += population_after_selection

            population = sorted(new_population, key=lambda elem: self.get_fitness(elem), reverse=True)[:population_size]

        return population[0]

    # Fitness function for an accompaniment
    def get_fitness(self, chromosome: list[Chord]):

        # Calculating the average pitch
        notes_sum = 0
        notes_count = 0
        for chord in chromosome:
            for note in chord.notes:
                notes_sum += note
                notes_count += 1

        average_note = notes_sum / notes_count

        # Count how many dissonant chords we have
        def count_dissonances_score() -> int:
            count = 0
            dissonant_intervals = [6, 1, 11]
            for note in self.core_melody_notes:
                beat = note[0]
                melody_note = note[1]
                gene = chromosome[beat]
                for chord_note in gene.notes:
                    if abs(melody_note % 12 - chord_note % 12) in dissonant_intervals:
                        count += 1
            return count

        # Gives points for consonant notes in chords to a respective melody one
        def count_consonances_score():
            count = 0
            perfect_consonance_score = 4
            imperfect_consonance_score = 1
            perfect_consonance_intervals = [0, 5, 7]
            imperfect_consonance_intervals = [2, 10, 3, 4, 8, 9]
            dissonant_intervals = [6, 1, 11]

            for note in self.core_melody_notes:
                has_dissonant_interval = False
                local_count = 0
                beat = note[0]
                melody_note = note[1]
                gene = chromosome[beat]
                for chord_note in gene.notes:
                    if abs(melody_note % 12 - chord_note % 12) in dissonant_intervals:
                        has_dissonant_interval = True
                        break
                    if abs(melody_note % 12 - chord_note % 12) in perfect_consonance_intervals:
                        local_count += perfect_consonance_score
                    if abs(melody_note % 12 - chord_note % 12) in imperfect_consonance_intervals:
                        local_count += imperfect_consonance_score
                if not has_dissonant_interval:
                    count += local_count

            return count

        # Counts the number of progressions found in accompaniment
        def count_progressions_score():
            count = 0
            progressions = [[0, 4, 5, 3],
                            [4, 5, 3, 0],
                            [5, 3, 0, 4],
                            [3, 0, 4, 5],
                            [0, 5, 3, 4],
                            [5, 3, 4, 0],
                            [3, 4, 0, 5],
                            [4, 0, 5, 3],
                            [1, 4, 0],
                            ]
            for progression in progressions:
                for idx in range(len(chromosome) - len(progression) + 1):

                    window = [self.steps_offsets.index(
                        abs(chord.main_note - self.tonic) % 12) if (abs(
                        chord.main_note - self.tonic) % 12) in self.steps_offsets else -1 for chord in
                              chromosome[idx: idx + len(progression)]]
                    if window == progression:
                        count += 1

            return count

        # Finds how many repetitive subarrays are there in accompaniment
        def count_repetitions_score():
            main_note_repetition_score = 1
            full_chord_repetition_score = 5
            window_size = 3

            count = 0

            for i in range(len(chromosome) - window_size + 1):
                main_note_repetition = True
                full_chord_repetition = True
                window = chromosome[i:i + window_size]
                current_chord = window[0]
                for chord in window[1:]:
                    main_note_repetition &= chord.main_note == current_chord.main_note
                    full_chord_repetition &= chord.notes == current_chord.notes
                    current_chord = chord
                if main_note_repetition:
                    count += main_note_repetition_score
                if full_chord_repetition:
                    count += full_chord_repetition_score
            return count

        # Counts how many notes are not further from average pitch than I want
        def count_octaves_difference_score():
            acceptance_range = 1
            count = 0

            for chord in chromosome:
                for note in chord.notes:
                    if abs(note - average_note) <= acceptance_range * 12:
                        count += 1
            return count

        # Counts how many pairs of neighboring chords violate the maximum pitch difference that I want
        def count_neighbors_maximum_offset_score():
            acceptance_range = 12
            count = 0
            for i in range(len(chromosome) - 1):
                difference = max(abs(min(chromosome[i].notes) - max(chromosome[i + 1].notes)),
                                 abs(max(chromosome[i].notes) - min(chromosome[i + 1].notes)))
                count += 1 if difference <= acceptance_range else 0
            return count

        # Counts how far away from the average of the melody and accompaniment pitches difference lies from the one I want
        def count_melody_accompaniment_pitch_difference_score():
            ideal_distance = 18

            difference = self.average_note - average_note

            return abs(ideal_distance - difference)

        def count_melody_gaps_accompaniment_repetitions_score():
            gaps_count = self.chords_number - len(self.core_melody_notes)
            beats_without_melody_note = set(range(self.chords_number))
            for beat, _ in self.core_melody_notes:
                beats_without_melody_note.remove(beat)

            count = 0

            for i in range(1, self.chords_number):
                if i in beats_without_melody_note and chromosome[i].notes == chromosome[i - 1].notes:
                    count += 1

            return abs(gaps_count - count)

        score_1 = count_consonances_score()
        score_2 = count_progressions_score()
        score_3 = -count_repetitions_score()
        score_4 = count_octaves_difference_score()
        score_5 = -count_dissonances_score()
        score_6 = count_neighbors_maximum_offset_score()
        score_7 = -count_melody_accompaniment_pitch_difference_score()
        score_8 = -count_melody_gaps_accompaniment_repetitions_score()

        consonance_coef = 1.8
        dissonance_coef = 500
        progression_coef = 10
        repetitions_coef = 2
        octaves_difference_coef = 3.1
        neighbors_maximum_offset_coef = 10
        melody_accompaniment_pitch_difference_coef = 3.8 * self.chords_number / 16  # normalize the value as this fitness part does not depend on  length accompaniments unlike others
        melody_gaps_accompaniment_repetitions_coef = 20

        return consonance_coef * score_1 + progression_coef * score_2 + repetitions_coef * score_3 + octaves_difference_coef * score_4 + dissonance_coef * score_5 + neighbors_maximum_offset_coef * score_6 + melody_accompaniment_pitch_difference_coef * score_7 + melody_gaps_accompaniment_repetitions_coef * score_8


def parse_input(midi: mido.MidiFile):
    current_tick = 0
    notes = []
    notes_to_beat = []
    ticks_per_bit = midi.ticks_per_beat
    last_note_beat = None

    for message in midi.tracks[1]:
        current_tick += message.time
        if message.type != 'note_on' and message.type != 'note_off':
            continue

        if message.type == 'note_on':
            notes.append(message.note)
            if current_tick % ticks_per_bit == 0:
                notes_to_beat.append((current_tick // ticks_per_bit, message.note))
            last_note_beat = current_tick // ticks_per_bit

    return notes, notes_to_beat, last_note_beat + 1


def encode_output(accompaniment: list[Chord], melody: mido.MidiFile) -> mido.MidiFile:
    track = mido.MidiTrack()
    for chord in accompaniment:
        track += [mido.Message('note_on', note=note, velocity=45, time=0) for
                  note in chord.notes]
        track += [mido.Message('note_off', note=chord.notes[0], velocity=45, time=melody.ticks_per_beat)]
        track += [mido.Message('note_off', note=note, velocity=45, time=0) for
                  note in chord.notes[1:]]

    output = copy.deepcopy(melody)
    output.tracks.append(track)
    return output


file_path = 'input.mid'
melody = mido.MidiFile(file_path)

all_notes, core_notes, beat_number = parse_input(melody)
key: music21.key.Key = music21.converter.parse(file_path).analyze('key')

genetic_algorithm = GA(all_notes, core_notes, key, beat_number)

accompaniment = genetic_algorithm.run(population_size=300, iterations=500, selection_rate=0.5,
                                      crossover_rate=0.4, gene_mutation_probability=0.05)

encode_output(accompaniment, melody).save("output.mid")
