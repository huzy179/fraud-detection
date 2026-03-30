FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

ENV NODE_ENV=production
ENV NEXT_PUBLIC_API_URL=http://api:8000

EXPOSE 3000

CMD ["npm", "run", "dev"]
