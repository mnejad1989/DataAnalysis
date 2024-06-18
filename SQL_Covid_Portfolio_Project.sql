select count(*) from PortfolioSQLProject..CovidDeaths;

select * 
from PortfolioSQLProject..CovidDeaths
order by 3,4
-- order by location and date

select *
from PortfolioSQLProject..CovidVaccinations
order by 3,4


-- select the data that we are going to use

select location,date,total_cases, new_cases, total_deaths, population
from PortfolioSQLProject..CovidDeaths
order by 1,2

-- Examining total cases vs Total Deaths

select location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as Death_Percentage
from PortfolioSQLProject..CovidDeaths
order by location,date desc

-- in france
select location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as Death_Percentage
from PortfolioSQLProject..CovidDeaths
where location like 'france'
order by location,date desc

-- total cases vs population

select location, date, total_cases, population, (total_cases/nullif(population,0))*100 as infection_rate
from PortfolioSQLProject..CovidDeaths
where location like '%france%'
order by date desc

-- country with the highest infection rate and verifying the time the query takes to run
 
 set statistics time on

 select location, max(total_cases) as latest_infection_count, population, (max(total_cases)/nullif(population,0))*100 as infection_rate
 from PortfolioSQLProject..CovidDeaths
 group by location, population
 order by infection_rate desc
 
 set statistics time off

 -- how many people died and its percentage rate againt the population

 select location, population, max(total_deaths) as latest_deaths_count,(MAX(total_deaths)/nullif(population,0))*100 as death_population_rate
 from PortfolioSQLProject..CovidDeaths
 where continent is not null
 group by location,population
 order by death_population_rate desc

 --sometimes the continent is null and it is written in the location section
 --that's why I have added  where continent is not null


  -- how many people died in each continent and its percentage rate againt the population

select continent, SUM(total_deaths) as Death,sum(population) as Population
from (select continent,location,population,total_deaths,
	ROW_NUMBER() over (partition by location order by date desc) as rn
from PortfolioSQLProject..CovidDeaths) as LatestRecords
where rn=1 and continent is not null
group by continent

-- continets with the highest death/population rates

select continent, sum(total_deaths) as TotalDeaths,sum(population) as TotalPopulation,
(sum(total_deaths)/sum(population))*100 as DeathPopulationRates
from (select continent,location, total_deaths, population,
	ROW_NUMBER() over(partition by location order by date desc) as rn
	from PortfolioSQLProject..CovidDeaths) as LatestRecord
where rn=1 and continent is not null
group by continent
order by DeathPopulationRates desc


-- Global numbers
select sum(total_cases) As Total_Cases,sum(total_deaths) AS Total_Deaths,
	sum(population) as Population, (sum(total_deaths)/sum(total_cases)) *100 as DeathRate
from (select location,max(population) as population,max(total_cases)As total_cases,max(total_deaths) as total_deaths,max(date) as date
	from PortfolioSQLProject..CovidDeaths
	where continent is not null
	group by location
	) as LatestRecords


-- looking at the vaccination table

select * 
from PortfolioSQLProject..CovidVaccinations

-- join two tables
select *
from PortfolioSQLProject..CovidDeaths de
join PortfolioSQLProject..CovidVaccinations va
	on de.location = va.location and de.date = va.date

-- Looking at total population vs vaccination
select de.location,de.population,max(va.total_vaccinations) Vaccinations 
from PortfolioSQLProject..CovidDeaths de
join PortfolioSQLProject..CovidVaccinations va
	on de.location = va.location and de.date = va.date
where de.continent is not null
group by de.location, de.population
order by location

-- Using partitiion. Looking at total population, new and total vaccination by date
select de.location,de.population,de.date,va.new_vaccinations, 
sum(va.new_vaccinations) over (partition by de.location order by de.location, de.date) TotalVaccinationToTheDate 
from PortfolioSQLProject..CovidDeaths de
join PortfolioSQLProject..CovidVaccinations va
	on de.location = va.location and de.date = va.date
where de.continent is not null and va.new_vaccinations is not null
order by de.location, de.date

-- Using CTE. % of total vaccination by date
with PopVsVac(location,population,date,new_vaccinations, TotalVaccinationToTheDate)
as(
select de.location,de.population,de.date,va.new_vaccinations, 
sum(va.new_vaccinations) over (partition by de.location order by de.location, de.date) TotalVaccinationToTheDate 
from PortfolioSQLProject..CovidDeaths de
join PortfolioSQLProject..CovidVaccinations va
	on de.location = va.location and de.date = va.date
where de.continent is not null and va.new_vaccinations is not null)
select *,(TotalVaccinationToTheDate/population)*100 VaccinationToTheDate
from PopVsVac
order by location,date


-- Using Temp Table. % of total vaccination by date
drop table if exists #PercentPopVaccinated
Create table #PercentPopVaccinated
(location nvarchar(100),population numeric,date datetime,new_vaccinations numeric,
TotalVaccinationToTheDate numeric)

select * from #PercentPopVaccinated

insert into #PercentPopVaccinated
select de.location,de.population,de.date,va.new_vaccinations, 
sum(va.new_vaccinations) over (partition by de.location order by de.location, de.date) TotalVaccinationToTheDate 
from PortfolioSQLProject..CovidDeaths de
join PortfolioSQLProject..CovidVaccinations va
	on de.location = va.location and de.date = va.date
where de.continent is not null and va.new_vaccinations is not null

select *,(TotalVaccinationToTheDate/population)*100 VaccinationToTheDate
from #PercentPopVaccinated
order by location,date

-- creating views to store data for later visualizations
USE PortfolioSQLProject;
GO

create view PercentPopVaccinated as
select de.location,de.population,de.date,va.new_vaccinations, 
sum(va.new_vaccinations) over (partition by de.location order by de.location, de.date) TotalVaccinationToTheDate 
from PortfolioSQLProject..CovidDeaths de
join PortfolioSQLProject..CovidVaccinations va
	on de.location = va.location and de.date = va.date
where de.continent is not null and va.new_vaccinations is not null

Go

SELECT * FROM sys.views WHERE name = 'PercentPopVaccinated';






