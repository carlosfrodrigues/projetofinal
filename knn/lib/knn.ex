defmodule Knn.Point do
  defstruct [
    fields: nil,
    class: nil
  ]
end

defmodule Knn do
  alias NimbleCSV.RFC4180, as: CSV
  #alias LearnKit.Knn
  use Rustler, otp_app: :knn, crate: "knn"


  def predict(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def main do
    filewriter = fn (filename, data) -> 
      File.open(filename, [:append]) 
      |> elem(1) 
      |> IO.binwrite(data) 
    end
    
    {train, test} = get_parsed_CSV()
    Enum.each(0..199, fn _ ->

      value = measure(fn -> 
        predict(train, test)
      end)
      filewriter.("knn_rust.txt",  to_string(value) <> "\n") 
  end)
  end

  def measure(function) do
    function
    |> :timer.tc
    |> elem(0)
    |> Kernel./(1_000_000)

  end
  def get_parsed_CSV() do
    train = 
    "../trainingsample.csv" |> File.stream! |> CSV.parse_stream
    |> Enum.map(fn [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
    v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27,
    v28, amount, class] ->
      %Knn.Point{fields: [String.to_float(v1), String.to_float(v2), String.to_float(v3), String.to_float(v4),
        String.to_float(v5), String.to_float(v6), String.to_float(v7), elem(Float.parse(time), 0),
        String.to_float(v8), String.to_float(v9), String.to_float(v10),
        String.to_float(v11), String.to_float(v12), String.to_float(v13),
        String.to_float(v14), String.to_float(v15), String.to_float(v16),
        String.to_float(v17), String.to_float(v18), String.to_float(v19),
        String.to_float(v20), String.to_float(v21), String.to_float(v22),
        String.to_float(v23), String.to_float(v24), String.to_float(v25),
        String.to_float(v26), String.to_float(v27), String.to_float(v28),
        elem(Float.parse(amount), 0)
      ], class: String.to_integer(class)}
    end)

    test = 
    "../validationsample.csv" |> File.stream! |> CSV.parse_stream
    |> Enum.map(fn [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
    v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27,
    v28, amount, class] ->
      %Knn.Point{fields: [String.to_float(v1), String.to_float(v2), String.to_float(v3), String.to_float(v4),
        String.to_float(v5), String.to_float(v6), String.to_float(v7), elem(Float.parse(time), 0),
        String.to_float(v8), String.to_float(v9), String.to_float(v10),
        String.to_float(v11), String.to_float(v12), String.to_float(v13),
        String.to_float(v14), String.to_float(v15), String.to_float(v16),
        String.to_float(v17), String.to_float(v18), String.to_float(v19),
        String.to_float(v20), String.to_float(v21), String.to_float(v22),
        String.to_float(v23), String.to_float(v24), String.to_float(v25),
        String.to_float(v26), String.to_float(v27), String.to_float(v28),
        elem(Float.parse(amount), 0)
      ], class: String.to_integer(class)}
    end)

  {train, test}

  end

end