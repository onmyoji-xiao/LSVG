import argparse
import os

from dataset.scannet_scan import ScannetDataset, ScannetScan
import multiprocessing as mp
import pickle
from tqdm import tqdm


def scannet_loader(scan_ids, scannet, align, sample, box_pred):
    """Helper function to load the scans in memory.
    :param scan_id:
    :return: the loaded scan.
    """
    with open('./scannet/scannetv2_val.txt') as fs:
        val_lines = fs.readlines()
        val_ids = [l.strip() for l in val_lines]
    res = []
    for scan_id in tqdm(scan_ids):
        scan_i = ScannetScan(scan_id, scannet, align, load_semantic_label=True, sample=sample)
        # scan_i.load_all_objects_by_maskcluster()
        # scan_i.load_all_objects_by_pg()
        if box_pred > 0 and scan_id in val_ids:
            scan_i.load_all_objects_by_pg()
        else:
            scan_i.load_point_clouds_of_all_objects()
        res.append(scan_i)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReferIt3D')

    parser.add_argument('-scans_dir', type=str, default='/home/xf/codes/DataSet/ScanNet_PC',
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('-save_dir', type=str, default='../../scannet',
                        help='the path of the directory to be saved preprocessed scans as a .pkl')

    # Optional arguments.
    parser.add_argument('--n-processes', default=4, type=int,
                        help='the number of processes, -1 means use the available max')
    parser.add_argument('--process-only-zero-view', default=1, type=int, help='1: only 00_view of scans are used')
    parser.add_argument('--apply-global-alignment', default=1, type=int,
                        help='rotate/translate entire scan globally to aligned it with other scans')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--box_pred', type=int, default=1)
    args = parser.parse_args()

    scan_ids = sorted(os.listdir(args.scans_dir))

    if args.process_only_zero_view:
        scan_ids = [x for x in scan_ids if x.endswith('_00')]
        filename = 'scannet_00_views.pkl'
    else:
        scan_ids = [x for x in scan_ids if not x.endswith('_00')]
        filename = 'scannet_0x_views.pkl'

    # Prepare ScannetDataset
    idx_to_semantic_class_file = './mappings/scannet_idx_to_semantic_class.json'
    instance_class_to_semantic_class_file = './mappings/scannet_instance_class_to_semantic_class.json'
    axis_alignment_info_file = './scannet/scans_axis_alignment_matrices.json'

    scannet = ScannetDataset(args.scans_dir,
                             idx_to_semantic_class_file,
                             instance_class_to_semantic_class_file,
                             axis_alignment_info_file=axis_alignment_info_file)

    n_items = len(scan_ids)
    print('scan number:', n_items)
    if args.n_processes == -1:
        n_processes = min(mp.cpu_count(), n_items)
    else:
        n_processes = args.n_processes
    pool = mp.Pool(n_processes)
    batch = n_items // n_processes

   # scannet_loader(scan_ids, scannet, args.apply_global_alignment, args.sample, args.box_pred)
    result = []
    for i in range(n_processes):
        if i == n_processes - 1:
            result.append(pool.apply_async(func=scannet_loader,
                                           args=(scan_ids[batch * i:], scannet, args.apply_global_alignment,
                                                 args.sample, args.box_pred)))
        else:
            result.append(pool.apply_async(func=scannet_loader,
                                           args=(scan_ids[batch * i:batch * (i + 1)], scannet,
                                                 args.apply_global_alignment, args.sample, args.box_pred)))

    pool.close()
    pool.join()

    all_scans = []
    for res in result:
        all_scans.extend(res.get())
    #
    print('{} scans are writing...'.format(len(all_scans)))
    os.makedirs(args.save_dir, exist_ok=True)

    if args.box_pred > 0:
        filename = 'pg_'+filename
    with open(os.path.join(args.save_dir, filename), "wb") as fp_data:
        pickle.dump(all_scans, fp_data)
