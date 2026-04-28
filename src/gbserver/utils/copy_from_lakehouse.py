# pip install getdaft
# python gbserver/environment/copy_from_lakehouse.py general_english_sft_replay_small_1104 instruct_tune.replay tmp
import sys


def copy_from_lakehouse(datasetname, namespace, output_path):
    import daft
    from lakehouse import Table

    from gbserver.utils.lakehouse_utils import create_lakehouse_iceberg

    lh = create_lakehouse_iceberg()
    table = Table(lh=lh, table_name=datasetname, namespace=namespace)
    ibt = table.iceberg_table
    df = daft.read_iceberg(ibt)
    # df.show()
    # df.count_rows()
    df.write_parquet(output_path)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        datasetname = sys.argv[1]
        namespace = sys.argv[2]
        output_path = sys.argv[3]
        copy_from_lakehouse(datasetname, namespace, output_path)
        print(
            f"datasetname={datasetname} namespace={namespace} output_path={output_path}"
        )
    else:
        print(
            f"{sys.argv[0]}: Please provide three arguments: datasetname namespace output_path"
        )
