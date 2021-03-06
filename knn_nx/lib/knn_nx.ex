defmodule KnnNx do
  alias NimbleCSV.RFC4180, as: CSV
  import Nx.Defn

  def main do

    filewriter = fn (filename, data) -> 
      File.open(filename, [:append]) 
      |> elem(1) 
      |> IO.binwrite(data) 
    end

    {x_train, y_train, x_test, _y_test} = get_parsed_CSV()
    x_input = Nx.tensor(Enum.at(x_test, 0), type: {:f, 64})
    x_train_tensor = Nx.tensor(x_train, type: {:f, 64})
    Enum.each(0..199, fn _ ->
      value = measure(fn ->
        indices = predictNx1(x_train_tensor, x_input, 5)
        IO.inspect(indices.type)
        predictNx2(Nx.to_flat_list(indices), y_train)
      end)
      filewriter.("knn_nx.txt",  to_string(value) <> "\n") 
    end)

    Enum.each(0..199, fn _ ->
      value = measure(fn ->
        predict(x_train, y_train, Enum.at(x_test, 0), 5)
      end)
      filewriter.("knn.txt",  to_string(value) <> "\n")
    end)
    
  end
  
  @defn_compiler {EXLA, client: :default}
  defn euclideanNx(x_train, x_input) do
    Nx.subtract(x_train, x_input) |> Nx.map(fn point_train ->
        Nx.sqrt(Nx.sum(Nx.power(point_train, 2)))
      end)
  end

  @defn_compiler {EXLA, client: :default}
  defn predictNx1(x_train, x_input, k \\ 3) do
    distances =
    Nx.subtract(x_train, x_input) |> Nx.LinAlg.norm(axes: [1])
    Nx.argsort(distances)[0..k-1]
  end

  def predictNx2(indices, y_train) do
    values = Enum.map(indices, fn index ->
      Enum.at(y_train,index)
    end)
    mode(values) 
  end


  def mode(list) when is_list(list) do
    h = hist(list)
    max = Map.values(h) |> Enum.max()
    h |> Enum.find(fn {_, val} -> val == max end) |> elem(0)
  end

  def hist(list) when is_list(list) do
    list
    |> Enum.reduce(%{}, fn tag, acc -> Map.update(acc, tag, 1, &(&1 + 1)) end)
  end

  def euclidean(p1, p2) do
    Enum.zip(p1, p2)
    |> Enum.reduce(0, fn {xi, yi}, acc ->
        acc + :math.pow(xi - yi, 2)
      end)
    |> :math.sqrt()
  end

  def predict(x_train, y_train, x_input, k) do
    distances =
      x_train |> Enum.map(fn point_train -> 
        euclidean(point_train, x_input)
      end)
    
    indices =
    distances |> Enum.with_index()
    |> Enum.sort_by(fn {val, _idx} -> val end)
    |> Enum.take(k)
    |> Enum.map(fn {_, idx} -> idx end)

    indices |> Enum.map(fn idx -> Enum.at(y_train, idx) end) |> mode()
  end


  def measure(function) do
    function
    |> :timer.tc
    |> elem(0)
    |> Kernel./(1_000_000)
  end

  def get_parsed_CSV() do
    #y_train = []
    x_train = 
    "../trainingsample.csv" |> File.stream! |> CSV.parse_stream
    |> Enum.map(fn [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
    v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27,
    v28, amount, _class] ->
      [String.to_float(v1), String.to_float(v2), String.to_float(v3), String.to_float(v4),
        String.to_float(v5), String.to_float(v6), String.to_float(v7), elem(Float.parse(time), 0),
        String.to_float(v8), String.to_float(v9), String.to_float(v10),
        String.to_float(v11), String.to_float(v12), String.to_float(v13),
        String.to_float(v14), String.to_float(v15), String.to_float(v16),
        String.to_float(v17), String.to_float(v18), String.to_float(v19),
        String.to_float(v20), String.to_float(v21), String.to_float(v22),
        String.to_float(v23), String.to_float(v24), String.to_float(v25),
        String.to_float(v26), String.to_float(v27), String.to_float(v28),
        elem(Float.parse(amount), 0)
      ]
    end)

    y_train = 
    "../trainingsample.csv" |> File.stream! |> CSV.parse_stream
    |> Enum.map(fn [_time, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, 
    _v10, _v11, _v12, _v13, _v14, _v15, _v16, _v17, _v18, _v19, _v20, _v21, _v22,
    _v23, _v24, _v25, _v26, _v27, _v28, _amount, class] ->
      [String.to_integer(class)]
    end)

    #y_test = []
    x_test = 
    "../validationsample.csv" |> File.stream! |> CSV.parse_stream
    |> Enum.map(fn [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, 
    v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27,
    v28, amount, _class] ->
      [String.to_float(v1), String.to_float(v2), String.to_float(v3), String.to_float(v4),
        String.to_float(v5), String.to_float(v6), String.to_float(v7), elem(Float.parse(time), 0),
        String.to_float(v8), String.to_float(v9), String.to_float(v10),
        String.to_float(v11), String.to_float(v12), String.to_float(v13),
        String.to_float(v14), String.to_float(v15), String.to_float(v16),
        String.to_float(v17), String.to_float(v18), String.to_float(v19),
        String.to_float(v20), String.to_float(v21), String.to_float(v22),
        String.to_float(v23), String.to_float(v24), String.to_float(v25),
        String.to_float(v26), String.to_float(v27), String.to_float(v28),
        elem(Float.parse(amount), 0)
      ]
    end)

    y_test = 
    "../validationsample.csv" |> File.stream! |> CSV.parse_stream
    |> Enum.map(fn [_time, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, 
    _v10, _v11, _v12, _v13, _v14, _v15, _v16, _v17, _v18, _v19, _v20, _v21, _v22,
    _v23, _v24, _v25, _v26, _v27, _v28, _amount, class] ->
      [String.to_integer(class)]
    end)

    {x_train, y_train, x_test, y_test}  
  end
end
