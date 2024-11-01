function [inDegreeList] = fct_inDegreeList(inDegreeList, NNeure,NNeuri,sumKinExc,sumKinInh, max_value, min_value)

% Calculate the current sum and the target sum
current_sumExc = sum(inDegreeList(1:NNeure));
current_sumInh = sum(inDegreeList(1+NNeure:end));

% Step 2: Adjust the inDegreeList(1:NNeure) to match the target sum
if current_sumExc ~= sumKinExc
    adjustment = sumKinExc - current_sumExc;
    adjustment_per_sample = adjustment / NNeure;
    adjusted_samples = inDegreeList(1:NNeure) + adjustment_per_sample;

    % Ensure all values are integers and respect the max_value and min_value limit
    adjusted_samples = round(adjusted_samples);
    adjusted_samples = min(adjusted_samples, max_value);
    adjusted_samples = max(adjusted_samples, min_value);

    % Handle any rounding issues that might affect the total sum
    adjusted_sum = sum(adjusted_samples);
    if adjusted_sum ~= sumKinExc
        difference = sumKinExc - adjusted_sum;
        for i = 1:abs(difference)
            idx = mod(i - 1, NNeure) + 1;
            if adjusted_sum < sumKinExc && adjusted_samples(idx) < max_value
                adjusted_samples(idx) = adjusted_samples(idx) + 1;
            elseif adjusted_sum > sumKinExc && adjusted_samples(idx) > min_value
                adjusted_samples(idx) = adjusted_samples(idx) - 1;
            end
        end
    end
end
inDegreeList(1:NNeure)=adjusted_samples;

% Step 3: Adjust the inDegreeList(1+NNeure:end) to match the target sum
if current_sumInh ~= sumKinInh
    adjustment = sumKinInh - current_sumInh;
    adjustment_per_sample = adjustment / NNeuri;
    adjusted_samples = inDegreeList(1+NNeure:end) + adjustment_per_sample;

    % Ensure all values are integers and respect the max_value limit
    adjusted_samples = round(adjusted_samples);
    adjusted_samples = min(adjusted_samples, max_value);

    % Handle any rounding issues that might affect the total sum
    adjusted_sum = sum(adjusted_samples);
    if adjusted_sum ~= sumKinInh
        difference = sumKinInh - adjusted_sum;
        for i = 1:abs(difference)
            idx = mod(i - 1, NNeuri) + 1;
            if adjusted_sum < sumKinInh && adjusted_samples(idx) < max_value
                adjusted_samples(idx) = adjusted_samples(idx) + 1;
            elseif adjusted_sum > sumKinInh && adjusted_samples(idx) > 0
                adjusted_samples(idx) = adjusted_samples(idx) - 1;
            end
        end
    end
end
inDegreeList(1+NNeure:end)=adjusted_samples;
end