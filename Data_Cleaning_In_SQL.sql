Select *
from PortfolioSQLProject..NashvilleHousingData

-- Standardize Date Format

Select SaleDate
from PortfolioSQLProject..NashvilleHousingData

Alter table PortfolioSQLProject..NashvilleHousingData
Alter column SaleDate date

Select SaleDate
from PortfolioSQLProject..NashvilleHousingData



-- Populate Property Address Data

Select count(*) as NullPropertyAddresses
from PortfolioSQLProject..NashvilleHousingData
where PropertyAddress is null

Select *
from PortfolioSQLProject..NashvilleHousingData
where PropertyAddress is null

-- ParcelIds have the same Property Address
Select *
from PortfolioSQLProject..NashvilleHousingData
order by ParcelID

-- replace null values with the address of other rows with the same ParcelID
Update PortfolioSQLProject..NashvilleHousingData
Set PropertyAddress = (
	select top 1 PropertyAddress
	from PortfolioSQLProject..NashvilleHousingData as sub
	where sub.PropertyAddress is not null 
	and ParcelID=sub.ParcelID
)
Where PropertyAddress is null

-- check the replacement
Select count(*) as NullPropertyAddresses
from PortfolioSQLProject..NashvilleHousingData
where PropertyAddress is null

-- Breaking out Address into Individual Columns (Address, City, State)
-- Using substring

--PropertyAddress
Select PropertyAddress
from PortfolioSQLProject..NashvilleHousingData

Alter table PortfolioSQLProject..NashvilleHousingData
drop column PostalCode 

Alter table PortfolioSQLProject..NashvilleHousingData
add PropertyCity nvarchar(150)

update PortfolioSQLProject..NashvilleHousingData
set PropertyCity = substring(PropertyAddress,CHARINDEX(',',PropertyAddress)+1, len(PropertyAddress))

update PortfolioSQLProject..NashvilleHousingData
set PropertyAddress = substring(PropertyAddress,1,CHARINDEX(',',PropertyAddress)-1)


-- Owner address
-- Breaking Down using Parcename

select OwnerAddress 
from PortfolioSQLProject..NashvilleHousingData


select 
parsename(replace(OwnerAddress, ',','.'),3) as OwnerAddress,
PARSENAME(replace(OwnerAddress, ',','.'),2) as OwnerCity,
PARSENAME(replace(OwnerAddress, ',','.'),1) as OwnerState
from PortfolioSQLProject..NashvilleHousingData

Alter table PortfolioSQLProject..NashvilleHousingData
add OwnerCity nvarchar(150)


Alter table PortfolioSQLProject..NashvilleHousingData
add OwnerState nvarchar(150)

update PortfolioSQLProject..NashvilleHousingData
set OwnerCity = PARSENAME(replace(OwnerAddress, ',','.'),2)

update PortfolioSQLProject..NashvilleHousingData
set OwnerState = PARSENAME(replace(OwnerAddress, ',','.'),1)

update PortfolioSQLProject..NashvilleHousingData
set OwnerAddress = PARSENAME(replace(OwnerAddress, ',','.'),3)


-- Change Y and N to Yes and No in "SoldAsVacant" field
-- Verifying the column's values
select distinct(SoldAsVacant),count(SoldAsVacant) as Count
from PortfolioSQLProject..NashvilleHousingData
Group by SoldAsVacant
Order by 2


-- update and replace the values
update PortfolioSQLProject..NashvilleHousingData
set SoldAsVacant = Case when SoldAsVacant = 'Y' then 'Yes'
						when SoldAsVacant = 'N' then 'No'
						else SoldAsVacant
						end

-- verify the values
select distinct(SoldAsVacant)
from PortfolioSQLProject..NashvilleHousingData


-- Remove the duplicates (in this scenario we need it)

-- Viewing the duplicates
With RowNumCTE as(
Select *,
	ROW_NUMBER() over (
		partition by ParcelID,PropertyAddress, SaleDate, SalePrice, LegalReference
		order by UniqueID
	) as RN
from PortfolioSQLProject..NashvilleHousingData)
select * from RowNumCTE
where RN>1
order by ParcelID


-- Deleting the duplicates
With RowNumCTE as(
Select *,
	ROW_NUMBER() over (
		partition by ParcelID,PropertyAddress, SaleDate, SalePrice, LegalReference
		order by UniqueID
	) as RN
from PortfolioSQLProject..NashvilleHousingData)
Delete from RowNumCTE
where RN>1


-- Delete unused columns
-- It's just for show case. Usually We don't do this to raw data
-- We should be cautious
-- For example we can presume that we don't need TaxDistrict

Alter table PortfolioSQLProject..NashvilleHousingData
drop column TaxDistrict

select *
from PortfolioSQLProject..NashvilleHousingData



