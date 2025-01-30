import json
from PyBioMed import Pyprotein
from jax_unirep import get_reps
from jax_unirep.utils import load_params
import asyncio
import aiohttp
from domain_analysis import make_fasta


def get_sequences(accessions):
    timeout = aiohttp.ClientTimeout(total=3600)
    return_dict = {}

    async def gather_accessions():
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            accession_tasks = []

            for c, accession in enumerate(accessions):
                task = asyncio.create_task(get_sequence(session, accession))
                accession_tasks.append(task)
            await asyncio.gather(*accession_tasks)

    async def get_sequence(session, accession):
        url = f'https://rest.uniprot.org/uniprotkb/search?query=(accession:{accession})&fields=sequence'
        async with session.get(url=url, ssl=False) as response:
            response = await response.json()
            sequence = response["results"][0]['sequence']['value']
            return_dict[accession] = sequence

    asyncio.get_event_loop().run_until_complete(gather_accessions())
    return return_dict


def parse_ids(path):
    with open(path) as file:
        lines = file.readlines()
        table_dict = {k: v for v, k in enumerate(lines[0].strip().split('\t'))}
        accessions = set()
        for line in lines[1:]:
            accession = line.split()[table_dict['accession']]
            accessions.add(accession.strip())
    return sorted(list(accessions))


def get_descriptors(sequence_dict):
    return_dict = {}
    params = load_params(paper_weights=256)[1]

    async def gather_accessions():
        accession_tasks = []

        for c, accession in enumerate(sequence_dict.keys()):
            task = asyncio.create_task(get_existing_descriptor(accession, c))
            accession_tasks.append(task)
            # break
        await asyncio.gather(*accession_tasks)

    async def get_acc(accession, c, max_lag=30, z_scales_amount=3):
        z_scale = {'A':
                       [0.24,
                        -2.32,
                        0.60,
                        -0.14,
                        1.30],
                   'M':
                       [-2.85,
                        -0.22,
                        0.47,
                        1.94,
                        -0.98],
                   'C':
                       [0.84,
                        -1.67,
                        3.71,
                        0.18,
                        -2.65],
                   'N':
                       [3.05,
                        1.62,
                        1.04,
                        -1.15,
                        1.61],
                   'D':
                       [3.98,
                        0.93,
                        1.93,
                        -2.46,
                        0.75],
                   'P':
                       [-1.66,
                        0.27,
                        1.84,
                        0.70,
                        2.00],
                   'E':
                       [3.11,
                        0.26,
                        -0.11,
                        -3.04,
                        -0.25],
                   'Q':
                       [1.75,
                        0.50,
                        -1.44,
                        -1.34,
                        0.66],
                   'F':
                       [-4.22,
                        1.94,
                        1.06,
                        0.54,
                        -0.62],
                   'R':
                       [3.52,
                        2.50,
                        -3.50,
                        1.99,
                        -0.17],
                   'G':
                       [2.05,
                        -4.06,
                        0.36,
                        -0.82,
                        -0.38],
                   'S':
                       [2.39,
                        -1.07,
                        1.15,
                        -1.39,
                        0.67],
                   'H':
                       [2.47,
                        1.95,
                        0.26,
                        3.90,
                        0.09],
                   'T':
                       [0.75,
                        -2.18,
                        -1.12,
                        -1.46,
                        -0.40],
                   'I':
                       [-3.89,
                        -1.73,
                        -1.71,
                        -0.84,
                        0.26],
                   'V':
                       [-2.59,
                        -2.64,
                        -1.54,
                        -0.85,
                        -0.02],
                   'K':
                       [2.29,
                        0.89,
                        -2.49,
                        1.49,
                        0.31],
                   'W':
                       [-4.36,
                        3.94,
                        0.59,
                        3.44,
                        -1.59],
                   'L':
                       [-4.28,
                        -1.30,
                        -1.49,
                        -0.72,
                        0.84],
                   'Y':
                       [-2.54,
                        2.44,
                        0.43,
                        0.04,
                        -1.47]}
        sequence = sequence_dict[accession]

        try:
            descriptor = []
            for first_z in range(z_scales_amount):
                for second_z in range(z_scales_amount):
                    for lag in range(1, max_lag + 1):
                        state = 0
                        for symbol in range(len(sequence) - lag):
                            state += z_scale[sequence[symbol]][first_z] * z_scale[sequence[symbol + lag]][second_z]
                        descriptor.append(state / (len(sequence) - lag))
            print(descriptor)
            print(len(descriptor))
            print(c)
            return_dict[accession] = descriptor
        except KeyError:
            print('селеноцистеин')
        except ZeroDivisionError:
            print(sequence)

    async def get_existing_descriptor(accession, c):
        sequence = sequence_dict[accession]
        try:
            # descriptor = [f for f in Pyprotein.PyProtein(sequence).GetPAAC(30).values()]
            descriptor, _, _ = get_reps(sequence, params=params, mlstm_size=256)
            print(list(descriptor[0]))
            print(len(list(descriptor[0])))
            print(c)
            return_dict[accession] = [float(i) for i in list(descriptor[0])]
        except KeyError:
            print('селеноцистеин')
        except ZeroDivisionError:
            print(sequence)

    asyncio.get_event_loop().run_until_complete(gather_accessions())
    return return_dict


def main():
    path = "C:\\Users\\georg\\OneDrive\\Документы\\лаба\\papyrus_all_wt_human.tsv"
    sequence_path = "C:\\Users\\georg\\OneDrive\\Документы\\лаба\\papyrus\\proteins\\sequences.json"
    descriptor_name = 'UniRep_256'
    desriptor_path = f"C:\\Users\\georg\\OneDrive\\Документы\\лаба\\papyrus\\proteins\\papyrus_{descriptor_name}.json"
    # accessions = parse_ids(path)
    # sequence_dict = get_sequences(accessions)
    # with open(sequence_path, 'w') as file:
    #     json.dump(sequence_dict, file)
    # with open(sequence_path) as file:
    #     sequence_dict = json.load(file)

    sequence_dict = make_fasta.parse_fasta(r'C:\Users\georg\OneDrive\Документы\лаба\proteases_3_kurs\proteins\Pr100.fa')
    print(sequence_dict)
    descriptor_dict = get_descriptors(sequence_dict)

    with open(desriptor_path, 'w') as file:
        json.dump(descriptor_dict, file)


if __name__ == '__main__':
    main()
