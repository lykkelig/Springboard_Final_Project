USE [MED_DATA]
GO

/****** Object:  Table [dbo].[PATIENT]    Script Date: 4/10/2024 1:12:45 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[PATIENT](
	[PATIENT_ID] [int] NOT NULL,
	[NAME] [varchar](128) NOT NULL,
	[DATE_OF_BIRTH] [date] NOT NULL,
	[GENDER] [char](1) NOT NULL,
	[NEXT_APPT] [date] NULL,
	[PDF_AVAILABLE] [char](1) NULL,
	[PDF_DIRECTORY] [varchar](256) NULL,
 CONSTRAINT [PK_PATIENT] PRIMARY KEY CLUSTERED 
(
	[PATIENT_ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO 